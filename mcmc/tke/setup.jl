using Statistics, Distributions, PyPlot, OrderedCollections, Optim, JLD2, FileIO, Printf
using OceanTurb, Dao, ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LESbrary = OrderedDict(
                   # Non-rotating
                    "kato, N²: 1e-7" => (
                       filename = "kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx256_Nz256_averages.jld2", 
                       rotating = false, 
                             N² = 1e-7, 
                          first = 5, 
                           last = 81),

                    #"kato, N²: 2e-7" => (
                    #   filename = "kato_phillips_Nsq2.0e-07_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                    #   rotating = false, 
                    #         N² = 2e-7, 
                    #      first = 5, 
                    #       last = nothing),

                    #"kato, N²: 5e-7" => (
                    #   filename = "kato_phillips_Nsq5.0e-07_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                    #   rotating = false, 
                    #         N² = 5e-7, 
                    #      first = 5, 
                    #       last = nothing),

                   "kato, N²: 1e-6" => (
                       filename = "kato_phillips_Nsq1.0e-06_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                       rotating = false, 
                             N² = 1e-6, 
                          first = 5, 
                           last = 61),

                   "kato, N²: 1e-5" => (
                       filename = "kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx256_Nz256_averages.jld2", 
                       rotating = false, 
                             N² = 1e-5, 
                          first = 11, 
                           last = 121),

                    "kato, N²: 1e-4" => (
                       filename = "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2", 
                       rotating = false, 
                             N² = 1e-4, 
                          first = 21, 
                           last = nothing),

                    # Rotating
                    "ekman, N²: 1e-7" => (
                       filename = "stress_driven_Nsq1.0e-07_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       rotating = true, 
                             N² = 1e-7, 
                          first = 11, 
                           last = 201),

                    "ekman, N²: 1e-6" => (
                       filename = "stress_driven_Nsq1.0e-06_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       rotating = true, 
                             N² = 1e-6, 
                          first = 11, 
                           last = 75),

                    "ekman, N²: 1e-5" => (
                       filename = "stress_driven_Nsq1.0e-05_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       rotating = true, 
                             N² = 1e-5, 
                          first = 11, 
                           last = nothing),

                    "ekman, N²: 1e-4" => (
                       filename = "stress_driven_Nsq1.0e-04_f1.0e-04_Qu4.0e-04_Nh128_Nz128_averages.jld2",
                       rotating = true, 
                             N² = 1e-4, 
                          first = 11, 
                           last = nothing),
                   )

#####
##### KPP functionality
#####

@free_parameters KPPWindMixingParameters CRi CSL Cτ

function init_kpp_calibration(dataname; 
                                            Δz = 4, 
                                            Δt = 10second, 
                                  first_target = 5, 
                                   last_target = nothing, 
                                        fields = (:T, :U), 
                              relative_weights = [1.0 for f in fields],
                                   mixingdepth = ModularKPP.LMDMixingDepth(),
                                      kprofile = ModularKPP.StandardCubicPolynomial(),
                              profile_analysis = ValueProfileAnalysis
                              )

    # Model and data
    datapath = joinpath(LESbrary_path, dataname)
    data = ColumnData(datapath)
    model = ModularKPPOptimization.ColumnModel(data, Δt, Δ=Δz, mixingdepth=mixingdepth, kprofile=kprofile)

    # Create loss function and negative-log-likelihood object
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target

    # Estimate weights based on maximum variance in the data
    max_variances = [max_variance(data, field, targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    # Create loss function and NegativeLogLikelihood
    loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights,
                        profile = profile_analysis(model.grid))
    nll = NegativeLogLikelihood(model, data, loss)

    # Initial state for optimization step
    default_parameters = DefaultFreeParameters(model, KPPWindMixingParameters)

    return nll, default_parameters
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
##### TKEMassFlux functionality
#####

function init_tke_calibration(dataname; 
                                              Δz = 4, 
                                              Δt = 10second, 
                                    first_target = 5, 
                                     last_target = nothing, 
                                          fields = (:T, :U, :e), 
                                relative_weights = [1.0 for f in fields],
                                   mixing_length = TKEMassFlux.SimpleMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                              eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                                profile_analysis = ValueProfileAnalysis(),
                                   unused_kwargs...
                              )


    # Model and data
    datapath = joinpath(LESbrary_path, dataname)
    data = ColumnData(datapath)
    model = TKEMassFluxOptimization.ColumnModel(data, Δt,
                                                                Δz = Δz,
                                                     mixing_length = mixing_length,
                                                    tke_wall_model = tke_wall_model,
                                                eddy_diffusivities = eddy_diffusivities
                                               )

    # Create loss function and negative-log-likelihood object
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target
    profile_analysis = on_grid(profile_analysis, model.grid)
    weights = estimate_weights(profile_analysis, data, fields, targets, relative_weights)
        
    # Create loss function and NegativeLogLikelihood
    loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights,
                        profile=profile_analysis)

    nll = NegativeLogLikelihood(model, data, loss)

    # Initial state for optimization step
    default_parameters = DefaultFreeParameters(model, TKEParametersToOptimize)

    return nll, default_parameters
end

set_bound!(bounds, name, bound) =
    name ∈ propertynames(bounds) && setproperty!(bounds, name, bound)

function get_bounds_and_variance(default_parameters)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters(((0.01, 2.0) for p in default_parameters)...)

    # Some special bounds, in the cases they are included.
    set_bound!(bounds, :Cᴷu,  (0.01, 0.2))
    set_bound!(bounds, :Cᴷe,  (0.01, 1.0))
    set_bound!(bounds, :Cᴷc,  (0.01, 1.0))
    set_bound!(bounds, :Cʷu★, (0.5, 6.0))

    variance = SomeFreeParameters((0.1 * (bound[2]-bound[1]) for bound in bounds)...)
    variance = Array(variance)

    return bounds, variance
end

function calibrate(nll, initial_parameters; samples=100, iterations=10,

                    annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+0,
                                                                        final_scale = 1e-2,
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

struct OptimSafeNegativeLogLikelihood{P, N} 
    parameters :: P
    negative_log_likelihood :: N
    function OptimSafeNegativeLogLikelihood(θ, ℒ)
        θ′ = deepcopy(θ)
        return new{typeof(θ′), typeof(ℒ)}(θ′, ℒ)
    end
end

function (nll::OptimSafeNegativeLogLikelihood)(θ) 
    for i in eachindex(nll.parameters)
        @inbounds nll.parameters[i] = θ[i]
    end
    return nll.negative_log_likelihood(nll.parameters)
end

"Use Optim to obtain a guess for optimal parameters."
function optim_optimized_parameters(nll, initial_parameters)
    optim_nll = OptimSafeNegativeLogLikelihood(initial_parameters, nll)
    residual = optimize(optim_nll, Array(initial_parameters))
    return typeof(initial_parameters)(residual.minimizer)
end

styles = ("--", ":", "-.", "o-", "^--")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Default kwargs for plot routines
default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8)
default_datakwargs = Dict(:linewidth=>3, :alpha=>0.6)
default_legendkwargs = Dict(:fontsize=>10, :loc=>"lower right", :frameon=>true, :framealpha=>0.5)

removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

"""
    visualize_results(data, model, params...)

Visualize the data alongside several realizations of `column_model`
for each set of parameters in `params`.
"""
function visualize_results(column_model, column_data, loss, param,
                                figsize = (16, 6),
                            paramlabels = ["" for p in params], datastyle="-",
                            modelkwargs = Dict(),
                             datakwargs = Dict(),
                           legendkwargs = Dict(),
                                 fields = (:U, :V, :T)
                          )

    # Merge defaults with user-specified options
     modelkwargs = merge(default_modelkwargs, modelkwargs)
      datakwargs = merge(default_datakwargs, datakwargs)
    legendkwargs = merge(default_legendkwargs, legendkwargs)

    #
    # Make plot
    #

    fig, axs = subplots(ncols=length(fields), figsize=figsize, sharey=true)
        
    set!(column_model, param)
    set!(column_model, column_data, loss.targets[1])
    run_until!(column_model.model, column_model.Δt, column_data.t[loss.targets[end]])

    lbl = @sprintf("Model, \$ t = %0.2f \$ hours", column_data.t[i]/hour)

    for (ipanel, field) in enumerate(fields)
        sca(axs[ipanel])
        model_field = getproperty(column_model.model.solution, field)
        plot(model_field, styles[1]; color=defaultcolors[iplot], label=lbl, modelkwargs...)
    end

    plot_data!(axs, column_data, loss.targets[[1, end]], fields; datastyle=datastyle, datakwargs...)
    format_axs!(axs, fields; legendkwargs...)

    return fig, axs
end

