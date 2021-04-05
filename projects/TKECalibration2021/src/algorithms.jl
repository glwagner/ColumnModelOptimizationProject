# Calibration algorithms
include("calibration/line_search_gradient_descent.jl")

set_if_present!(obj, name, field) = name ∈ propertynames(obj) && setproperty!(obj, name, field)

function get_bounds_and_variance(default_parameters)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters(((0.001, 6.0) for p in default_parameters)...)

    # Some special bounds, in the cases they are included.
    set_if_present!(bounds, :Cᴰ,  (0.01, 2.0))
    set_if_present!(bounds, :Cᴸᵇ, (0.01, 2.0))
    set_if_present!(bounds, :CʷwΔ, (0.01, 10.0))

    # Convective adjustment
    set_if_present!(bounds, :Cᴬ,  (0.01, 40.0))

    # Independent diffusivities
    set_if_present!(bounds, :Cᴷc, (0.005, 2.0))
    set_if_present!(bounds, :Cᴷe, (0.005, 0.5))

    # RiDependentDiffusivities
    set_if_present!(bounds, :CᴷRiᶜ, (-1.0, 2.0))
    set_if_present!(bounds, :CᴷRiʷ, (0.01, 10.0)) # (0.1, 2.0)
    set_if_present!(bounds, :Cᴷc⁺,  (0.001, 0.1))
    set_if_present!(bounds, :Cᴷe⁺,  (0.001, 0.1))
    set_if_present!(bounds, :Cᴷu⁺,  (0.001, 0.1))

    set_if_present!(bounds, :Cᴷc⁻,  (0.005, 0.5))
    set_if_present!(bounds, :Cᴷe⁻,  (0.005, 0.5))

    variances = SomeFreeParameters((0.02 * (bound[2] - bound[1]) for bound in bounds)...)

    variances = Array(variances)

    return bounds, variances
end

function get_bounds_and_variance(kpp_parameters::KPPWindMixingParameters)

    SomeFreeParameters = typeof(kpp_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters(((0.01, 2.0) for p in kpp_parameters)...)

    # Some special bounds, in the cases they are included.
    set_if_present!(bounds, :CSL, (0.01, 0.99))

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

function simulated_annealing(nll, initial_parameters; samples=100, iterations=10,

                    annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+0,
                                                                        final_scale = 1e-2,
                                                                   convergence_rate = 1.0,
                                                                    rate_adaptivity = 1.5),

                   covariance_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+0,
                                                                        final_scale = 1e+0,
                                                                   convergence_rate = 1.0,
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

function nelder_mead(nll, initial_parameters, ParametersToOptimize)
    ℒ(x) = nll(ParametersToOptimize(ℒ))
    r = Optim.optimize(ℒ, [initial_parameters...])
    params = ParametersToOptimize(Optim.minimizer(r))
    return params
end

function l_bfgs(nll, initial_parameters, ParametersToOptimize)
    ℒ(x) = nll(ParametersToOptimize(ℒ))
    r = Optim.optimize(ℒ, [initial_parameters...], LBFGS())
    params = ParametersToOptimize(Optim.minimizer(r))
    return params
end

function random_plugin(nll, initial_parameters, ParametersToOptimize; function_calls=1000)
    ℒ(x) = nll(ParametersToOptimize(x))
    bounds, _ = get_bounds_and_variance(initial_parameters)
    priors = [Uniform(b...) for b in bounds]
    method = RandomPlugin(priors, function_calls)
    minparam = optimize(ℒ, method; printresult=false)
    return ParametersToOptimize(minparam)
end

function gradient_descent(nll, initial_parameters, ParametersToOptimize; linebounds = (0, 100.0), linesearches = 100)
    ℒ(x) = nll(ParametersToOptimize(x))
    ∇loss(params) = gradient(ℒ, params) # numerical gradient
    method  = RandomLineSearch(linebounds = linebounds, linesearches = linesearches)
    bestparam = optimize(ℒ, ∇loss, [initial_parameters...], method);
    return ParametersToOptimize(bestparam)
end

# function calibrate_tke(datapath; initial_parameters=nothing, calibration_and_initialization_kwargs...)
#     nll, default_parameters = init_tke_calibration(datapath; calibration_and_initialization_kwargs...)
#     initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters
#
#     return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
# end
#
# function calibrate_kpp(datapath; initial_parameters=nothing, calibration_and_initialization_kwargs...)
#     nll, default_parameters = init_kpp_calibration(datapath; calibration_and_initialization_kwargs...)
#     initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters
#
#     return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
# end
#
# function batch_calibrate_tke(datapaths...; initial_parameters=nothing, calibration_and_initialization_kwargs...)
#
#     nll_list = []
#     for datapath in datapaths
#         nll, default_parameters = init_tke_calibration(datapath; calibration_kwargs...)
#         push!(nll_list, nll)
#     end
#
#     batched_nll = BatchedNegativeLogLikelihood([nll for nll in nll_list])
#
#     initial_parameters = initial_parameters === nothing ? default_parameters : initial_parameters
#
#     return calibrate(nll, initial_parameters; calibration_and_initialization_kwargs...)
# end

# buoyancy_frequency(data) = data.constants.g * data.constants.α * data.initial_conditions.dTdz
