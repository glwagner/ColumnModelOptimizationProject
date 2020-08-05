using Statistics, Distributions, PyPlot, OrderedCollections, JLD2, FileIO, Printf
using OceanTurb, Dao, ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization

@free_parameters TKEParametersToOptimize Cᴷu Cᴷc Cᴷe Cᴰ Cᴸʷ Cᴸᵇ Cʷu★ CʷwΔ
@free_parameters KPPWindMixingParameters CRi CSL Cτ
@free_parameters KPPWindMixingOrConvectionParameters CRi CSL Cτ Cb_U Cb_T

#####
##### The LESbrary (so to speak)
#####

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LESbrary = OrderedDict(
                   # Non-rotating
                    "kato, N²: 1e-7" => (
                       filename = "kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                       stressed = true,
                       rotating = false, 
                             N² = 1e-7, 
                          first = 5, 
                           last = 101),

                   "kato, N²: 1e-6" => (
                       filename = "kato_phillips_Nsq1.0e-06_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                       stressed = true,
                       rotating = false, 
                             N² = 1e-6, 
                          first = 5, 
                           last = 61),

                   "kato, N²: 1e-5" => (
                       filename = "kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx256_Nz256_averages.jld2", 
                       stressed = true,
                       rotating = false, 
                             N² = 1e-5, 
                          first = 11, 
                           last = 121),

                    "kato, N²: 1e-4" => (
                       filename = "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                       stressed = true,
                       rotating = false, 
                             N² = 1e-4, 
                          first = 21, 
                           last = nothing),

                    # Rotating
                    "ekman, N²: 1e-7" => (
                       filename = "stress_driven_Nsq1.0e-07_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true, 
                             N² = 1e-7, 
                          first = 11, 
                           last = 201),

                    "ekman, N²: 1e-6" => (
                       filename = "stress_driven_Nsq1.0e-06_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true, 
                             N² = 1e-6, 
                          first = 11, 
                           last = 75),

                    "ekman, N²: 1e-5" => (
                       filename = "stress_driven_Nsq1.0e-05_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true, 
                             N² = 1e-5, 
                          first = 11, 
                           last = nothing),

                    "ekman, N²: 1e-4" => (
                       filename = "stress_driven_Nsq1.0e-04_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true, 
                             N² = 1e-4, 
                          first = 11, 
                           last = 301),

                    # Convection
                    "convection, N²: 2e-6" => (
                       filename = "free_convection_Qb1.0e-07_Nsq2.0e-06_Nh256_Nz256_statistics.jld2",
                       stressed = false,
                       rotating = true, 
                             N² = 2e-6, 
                          first = 11, 
                           last = nothing),

                    "convection, N²: 1e-5" => (
                       filename = "free_convection_Qb1.0e-07_Nsq1.0e-05_Nh256_Nz256_statistics.jld2",
                       stressed = false,
                       rotating = true, 
                             N² = 1e-5, 
                          first = 31, 
                           last = nothing),

                   )

#####
##### KPP functionality
#####

kpp_fields(datum) = !(datum.stressed) ? (:T,) : !(datum.rotating) ? (:T, :U) : (:T, :U, :V)
kpp_relative_weights(datum) = !(datum.stressed) ? [1.0] : !(datum.rotating) ? [1.0, 1e-2] : [1.0, 1e-2, 1e-2]

tke_relative_weights(datum) = !(datum.stressed) ? [1.0, 1e-4] : !(datum.rotating) ? [1.0, 1e-2, 1e-4] : [1.0, 1e-2, 1e-2, 1e-4]
tke_fields(datum) = !(datum.stressed) ? (:T, :e) : !(datum.rotating) ? (:T, :U, :e) : (:T, :U, :V, :e)

"Initialize a calibration run for KPP."
function init_kpp_calibration(dataname; 
                                            Δz = 4, 
                                            Δt = 1second, 
                                  first_target = 5, 
                                   last_target = nothing, 
                                        fields = (:T, :U), 
                              relative_weights = [1.0 for f in fields],
                              profile_analysis = ValueProfileAnalysis(),
                              # KPP-specific kwargs:
                                   mixingdepth = ModularKPP.LMDMixingDepth(),
                                      kprofile = ModularKPP.StandardCubicPolynomial(),
                                 unused_kwargs...
                              )

    data = init_LESbrary_data(dataname)
    model = ModularKPPOptimization.ColumnModel(data, Δt, Δ=Δz, mixingdepth=mixingdepth, kprofile=kprofile)

    return init_negative_log_likelihood(model, data, first_target, last_target,
                                        fields, relative_weights, profile_analysis,
                                        KPPWindMixingOrConvectionParameters)
end


#####
##### TKEMassFlux functionality
#####

"Initialize a calibration run for the TKEMassFlux parameterization."
function init_tke_calibration(dataname; 
                                              Δz = 4, 
                                              Δt = 1second, 
                                    first_target = 5, 
                                     last_target = nothing, 
                                          fields = (:T, :U, :e), 
                                relative_weights = [1.0 for f in fields],
                                profile_analysis = ValueProfileAnalysis(),
                                # TKE-specific kwargs:
                                   mixing_length = TKEMassFlux.SimpleMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                              eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                                   unused_kwargs...
                              )

    data = init_LESbrary_data(dataname)
    model = TKEMassFluxOptimization.ColumnModel(data, Δt,
                                                                Δz = Δz,
                                                     mixing_length = mixing_length,
                                                    tke_wall_model = tke_wall_model,
                                                eddy_diffusivities = eddy_diffusivities
                                               )

    return init_negative_log_likelihood(model, data, first_target, last_target,
                                        fields, relative_weights, profile_analysis,
                                        TKEParametersToOptimize)
end

#####
##### Some utils common to KPP and TKEMassFlux
#####

function init_negative_log_likelihood(model, data, first_target, last_target,
                                      fields, relative_weights, profile_analysis,
                                      ParameterType)

    # Create loss function and negative-log-likelihood object
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target
    profile_analysis = on_grid(profile_analysis, model.grid)
    weights = estimate_weights(profile_analysis, data, fields, targets, relative_weights)
        
    # Create loss function and NegativeLogLikelihood
    loss = LossFunction(model, data, fields=fields, targets=targets, 
                        weights=weights, profile=profile_analysis)

    nll = NegativeLogLikelihood(model, data, loss)

    # Initial state for optimization step
    default_parameters = DefaultFreeParameters(model, ParameterType)

    return nll, default_parameters
end

function init_LESbrary_data(dataname)
    # Model and data
    datapath = joinpath(LESbrary_path, dataname)
    return ColumnData(datapath)
end

set_bound!(bounds, name, bound) = name ∈ propertynames(bounds) && setproperty!(bounds, name, bound)

function get_bounds_and_variance(default_parameters)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters(((0.01, 4.0) for p in default_parameters)...)

    # Some special bounds, in the cases they are included.
    set_bound!(bounds, :Cʷu★, (0.01, 10.0))

    variance = SomeFreeParameters((0.1 * (bound[2]-bound[1]) for bound in bounds)...)
    variance = Array(variance)

    return bounds, variance
end

function get_bounds_and_variance(kpp_parameters::KPPWindMixingParameters)

    SomeFreeParameters = typeof(kpp_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters(((0.01, 2.0) for p in kpp_parameters)...)

    # Some special bounds, in the cases they are included.
    set_bound!(bounds, :CSL,  (0.01, 0.99))

    variance = SomeFreeParameters((0.1 * (bound[2]-bound[1]) for bound in bounds)...)
    variance = Array(variance)

    return bounds, variance
end

function estimate_weights(profile::ValueProfileAnalysis, data, fields, targets, relative_weights)
    max_variances = [max_variance(data, field, targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    return weights
end

function estimate_weights(profile::GradientProfileAnalysis, data, fields, targets, relative_weights)
    gradient_weight = profile.gradient_weight
    value_weight = profile.value_weight

    @warn "Dividing the gradient weight of profile by height(data.grid) = $(height(data.grid))"
    gradient_weight = profile.gradient_weight = gradient_weight / height(data.grid)

    max_variances = [max_variance(data, field, targets) for field in fields]
    #max_gradient_variances = [max_gradient_variance(data, field, targets) for field in fields]

    weights = zeros(length(fields))
    for i in length(fields)
        σ = max_variances[i]
        #ς = max_gradient_variances[i]
        weights[i] = 1/σ * (value_weight + gradient_weight / height(data.grid))
    end

    if relative_weights != nothing
        weights .*= relative_weights
    end

    max_variances = [max_variance(data, field, targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    return weights
end

#####
##### Calibration
#####

function calibrate(nll, initial_parameters; samples=100, iterations=10,

                    annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+0,
                                                                        final_scale = 1e-3,
                                                                   convergence_rate = 1.0, 
                                                                    rate_adaptivity = 1.5),

                   covariance_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+0,
                                                                        final_scale = 1e+0,
                                                                   convergence_rate = 1e+0,
                                                                    rate_adaptivity = 1.0), 
                   unused_kwargs...
                   )

    bounds, variance = get_bounds_and_variance(initial_parameters)

    # Iterative simulated annealing...
    prob = anneal(nll, initial_parameters, variance, BoundedNormalPerturbation, bounds;
                           iterations = iterations,
                              samples = samples,
                   annealing_schedule = annealing_schedule,
                  covariance_schedule = covariance_schedule
                 )
    
    return prob
end

function calibrate_tke(datapath; initial_parameters=nothing, calibration_and_initialization_kwargs...)
    nll, default_parameters = init_tke_calibration(datapath; calibration_and_initialization_kwargs...)
    initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters

    return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
end

function calibrate_kpp(datapath; initial_parameters=nothing, calibration_and_initialization_kwargs...)
    nll, default_parameters = init_kpp_calibration(datapath; calibration_and_initialization_kwargs...)
    initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters

    return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
end

function batch_calibrate_tke(datapaths...; initial_parameters=nothing, calibration_and_initialization_kwargs...)

    nll_list = []
    for datapath in datapaths
        nll, default_parameters = init_tke_calibration(datapath; calibration_kwargs...)
        push!(nll_list, nll)
    end

    batched_nll = BatchedNegativeLogLikelihood([nll for nll in nll_list])

    initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters

    return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
end
  
buoyancy_frequency(data) = data.constants.g * data.constants.α * data.initial_conditions.dTdz 
