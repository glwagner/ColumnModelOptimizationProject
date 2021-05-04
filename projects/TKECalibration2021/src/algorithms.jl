# Calibration algorithms
include("calibration/line_search_gradient_descent.jl")

set_if_present!(obj, name, field) = name ∈ propertynames(obj) && setproperty!(obj, name, field)

# using Plots
# Plots.plot(LogNormal(0,1))
# rand(Normal(0,1),3)

function normalized_counts(normal; bin_width=0.05, bin_range=(-10,10))
    bins = [bin_width*x for x=Int(10/bin_range[1]):Int(10/bin_range[2])]
    counts = Dict(x => 0 for x in bins)

    for x in normal
        bin = ceil(x/bin_width)*bin_width
        if bin in keys(counts)
            counts[bin] += 1
        end
    end

    area = sum(values(counts))*bin_width
    normalized = Dict(x => 0.0 for x in bins)
    for (bin, count) in counts
        normalized[bin] = count/area
    end

    return normalized
end

# constrained_dist = Normal(7,1)
# lognormal = log.([x for x in rand(constrained_dist,1000000) if x>0.0]);
# lognormal_dist = fit(LogNormal, lognormal)
# normal = exp.(rand(lognormal_dist, 1000000));
#
#
# constrained_dist = Normal(0.001,0.1)
# plot(constrained_dist, label="N(μ=0.001, σ=0.1)")
#
# lognormal = [x for x in rand(constrained_dist,1000000) if x>0.0];
# constrained_dist = fit(LogNormal, lognormal)
# plot!(constrained_dist, label="lnN(μ=-2.93, σ=1.11)")
#
# normal = log.([x for x in rand(constrained_dist,1000000) if x>0.0]);
# normal_dist = fit(Normal, normal)
#
# julia> mean(constrained_dist)
# 0.09840050052320308
#
# julia> std(constrained_dist)
# 0.15290416640333576
#
#
# plot!(normal_dist, label="N(-0.71,0.21)")
#
# # lognormal = exp.(rand(normal_dist,1000000));
# # normal_dist = fit(Normal, lognormal)
#
#
#
# plot(constrained_dist, label = "normal N(7,1)")
# plot!(normalized_counts(lognormal, bin_width=0.05), label = "log.(normal N(7,1))")
# plot!(lognormal_dist, label = "lognormal N(0.657,0.078)")
# plot!(normalized_counts(normal, bin_width=0.05), label="exp.(lognormal N(0.657,0.078))", legend=:topright)
#
#
#
#
# constrained_dist = Normal(0.7,1)
# lognormal = exp.([x for x in rand(constrained_dist,1000000) if x>0.0]);
# lognormal_dist = fit(LogNormal, lognormal)
# normal = log.(rand(lognormal_dist, 1000000));
#
# plot(constrained_dist, label = "normal N(7,1)")
# plot!(normalized_counts(lognormal, bin_width=0.05), label = "log.(normal N(7,1))")
# plot!(lognormal_dist, label = "lognormal N(0.657,0.078)")
# plot!(normalized_counts(normal, bin_width=0.05), label="exp.(lognormal N(0.657,0.078))", legend=:topright)
#
#
#
#
#
# histogram(exp.rand(a, 100000))
#
# constrained_prior = fit(LogNormal, log.(rand(Normal(5,1),10000)))
#
#
# fit(LogNormal, rand(Normal(5,1),10000))
# exp(5 + 0.5)
#
# using StatsPlot

function get_bounds_and_variance(default_parameters; stds_within_bounds = 5)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters([(0.0, 10.0) for p in default_parameters]...)

    # Some special bounds, in the cases they are included.
    set_if_present!(bounds, :Cᴰ,   (0.0, 3.0))
    set_if_present!(bounds, :Cᴸᵇ,  (0.0, 7.5))
    set_if_present!(bounds, :CʷwΔ, (0.0, 40.0))
    set_if_present!(bounds, :Cʷu★, (0.0, 40.0))

    # Convective adjustment
    set_if_present!(bounds, :Cᴬ,   (0.0, 20.0))
    set_if_present!(bounds, :Cᴬu,  (0.0, 2.0))
    set_if_present!(bounds, :Cᴬc,  (0.0, 5.0))
    set_if_present!(bounds, :Cᴬe,  (0.0, 5.0))

    # Independent diffusivities
    set_if_present!(bounds, :Cᴷu, (0.0, 1.0))
    set_if_present!(bounds, :Cᴷc, (0.0, 1.0))
    set_if_present!(bounds, :Cᴷe, (0.0, 2.0))

    # RiDependentDiffusivities
    set_if_present!(bounds, :CᴷRiᶜ, (-1.0, 2.0))
    set_if_present!(bounds, :CᴷRiʷ, (0.0, 10.0)) # (0.1, 2.0)
    set_if_present!(bounds, :Cᴷc⁺,  (0.0, 0.1))
    set_if_present!(bounds, :Cᴷe⁺,  (0.0, 0.1))
    set_if_present!(bounds, :Cᴷu⁺,  (0.0, 0.1))

    set_if_present!(bounds, :Cᴷc⁻,  (0.0, 0.5))
    set_if_present!(bounds, :Cᴷe⁻,  (0.0, 0.5))

    # if stds_within_bounds = 3, 3 standard deviations to either side of the mean fits between the bounds
    variances = SomeFreeParameters((((bound[2] - bound[1])/(2 * stds_within_bounds))^2 for bound in bounds)...)

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
    max_variances = [mean_variance(data, field, targets) for field in fields]
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

function simulated_annealing(nll, initial_parameters;
                                                samples = 100,
                                             iterations = 5,
                  set_prior_means_to_initial_parameters = true,
                                     stds_within_bounds = 5,
                    annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+1,
                                                                        final_scale = 1e-2,
                                                                   convergence_rate = 1.0,
                                                                    rate_adaptivity = 1.5),

                   covariance_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+1,
                                                                        final_scale = 1e+0,
                                                                   convergence_rate = 1.0,
                                                                    rate_adaptivity = 1.0),
                   unused_kwargs...
                   )

    bounds, variance = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = set_prior_means_to_initial_parameters ? initial_parameters : [mean.(bounds)...]

    # Iterative simulated annealing...
    prob = anneal(nll, prior_means, variance, BoundedNormalPerturbation, bounds;
                           iterations = iterations,
                              samples = samples,
                   annealing_schedule = annealing_schedule,
                  covariance_schedule = covariance_schedule
                 )

    return prob
end

function nelder_mead(nll, initial_parameters)
    r = Optim.optimize(nll, [initial_parameters...])
    params = Optim.minimizer(r)
    return params
end

function l_bfgs(nll, initial_parameters)
    r = Optim.optimize(nll, [initial_parameters...], LBFGS())
    params = Optim.minimizer(r)
    return params
end

function random_plugin(nll, initial_parameters; function_calls=1000)
    bounds, _ = get_bounds_and_variance(initial_parameters)
    priors = [Uniform(b...) for b in bounds]
    method = RandomPlugin(priors, function_calls)
    minparam = optimize(nll, method; printresult=false)
    return minparam
end

function gradient_descent(nll, initial_parameters; linebounds = (0, 100.0), linesearches = 100)
    ∇loss(params) = gradient(nll, params) # numerical gradient
    method  = RandomLineSearch(linebounds = linebounds, linesearches = linesearches)
    bestparam = optimize(nll, ∇loss, [initial_parameters...], method);
    return bestparam
end

function ensemble_kalman_inversion(nll, initial_parameters;
                                                    set_prior_means_to_initial_parameters = true,
                                                    n_obs = 1,
                                                    noise_level = 1e-7,
                                                    N_ens = 100,
                                                    N_iter = 10,
                                                    stds_within_bounds = 5
                                                    )

    bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = set_prior_means_to_initial_parameters ? [initial_parameters...] : [mean.(bounds)...]

    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Independent noise for synthetic observations
    Γy = noise_level * Matrix(I, n_obs, n_obs)
    noise = MvNormal(zeros(n_obs), Γy)

    # Loss Function
    G(u) = nll(u)

    # Loss Function Minimum
    # y_obs  = G(u_star) .+ 0 * rand(noise)
    y_obs  = [0.0] .+ 0 * rand(noise)

    # Define Prior
    prior_distns = [Parameterized(Normal(prior_means[i], prior_variances[i])) for i in 1:length(bounds)]
    constraints = [[bounded(b...)] for b in [bounds...]]
    prior_names = String.([propertynames(initial_parameters)...])
    prior = ParameterDistribution(prior_distns, constraints, prior_names)
    prior_mean = reshape(get_mean(prior),:)
    prior_cov = get_cov(prior)

    # Calibrate
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())

    for i in 1:N_iter
        params_i = ekiobj.u[end]
        # g_ens = hcat([G(params_i.stored_data[:,i]) for i in 1:N_ens]...)'
        g_ens = hcat([G(params_i.stored_data[:,i]) for i in 1:N_ens]...)
        update_ensemble!(ekiobj, g_ens)
    end

    losses = [G([mean(ekiobj.u[i].stored_data, dims=2)...]) for i in 1:N_iter]

    A(i) = ekiobj.u[i].stored_data
    params = mean(ekiobj.u[end].stored_data, dims=2)
    mean_vars = [mean(sum((A(i) .- params).^2, dims=1)) for i in 1:N_iter]

    return params
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
