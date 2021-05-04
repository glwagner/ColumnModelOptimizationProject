relative_weight_options = Dict(
                "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0),
                "all_but_e" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 0.0),
                "all_uv" => Dict(:T => 0.0, :U => 1.0, :V => 1.0, :e => 0.0)
)

poptions = Dict(
    "TKEParametersConvectiveAdjustmentRiDependent" => TKEParametersConvectiveAdjustmentRiDependent,
    "TKEParametersConvectiveAdjustmentRiIndependent" => TKEParametersConvectiveAdjustmentRiIndependent,
    "TKEParametersRiIndependent" => TKEParametersRiIndependent,
    "TKEParametersRiDependent" => TKEParametersRiDependent
)

# function lognormal_μ_σ²(means, variances)
#     μs = []
#     σ²s = []
#     for i in 1:length(means)
#         k = variances[i]/(means[i]^2) + 1
#         μ = log(means[i]/sqrt(k))
#         σ² = log(k)
#         push!(μs, μ)
#         push!(σ²s, σ²)
#     end
#     return μs, σ²s
# end

function lognormal_μ_σ²(means, variances)
    k = variances ./ means.^2 .+ 1
    μs = log.(means ./ sqrt.(k))
    σ²s = log.(k)
    return μs, σ²s
end

"""
Arguments
nll: loss function
initial_parameters: where to begin the optimization

Keyword Arguments
set_prior_means_to_initial_parameters: whether to set the parameter prior means to the center of the parameter bounds or to given initial_parameters
n_obs: Number of synthetic observations from G(u)
noise_level: Observation noise level
N_ens: number of ensemble members, J
N_iter: number of EKI iterations, Ns
stds_within_bounds: number of (prior) standard deviations spanned by the parameter bounds (to either side of the mean)
"""
function eki(nll, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    n_obs = 1,
                                    noise_level = 10^(-1.0),
                                    N_ens = 10,
                                    N_iter = 20,
                                    stds_within_bounds = 0.6
                                    )

    # bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    # prior_means = set_prior_means_to_initial_parameters ? [initial_parameters...] : mean.(bounds)
    # # bounds, _ = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = [initial_parameters...]
    # μs, σ²s = lognormal_μ_σ²(prior_means, prior_variances)
    # println(μs, σ²s)
    # prior_distns = [Parameterized(Normal(μs[i], σ²s[i])) for i in 1:length(μs)]
    prior_distns = [Parameterized(Normal(0.0, stds_within_bounds)) for i in 1:length(bounds)]

    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Independent noise for synthetic observations
    Γy = noise_level * Matrix(I, n_obs, n_obs)
    # noise = MvNormal(zeros(n_obs), Γy)

    # Loss Function Minimum
    # y_obs  = G(u_star) .+ 0 * rand(noise)
    y_obs  = [0.0]

    # Define Prior
    # prior_distns = [Parameterized(Normal(prior_means[i], prior_variances[i])) for i in 1:length(bounds)]
    # prior_distns = [Parameterized(Normal(0.0,stds_within_bounds)) for i in 1:length(bounds)]
    # constraints = [[bounded(b...)] for b in [bounds...]]
    constraints = [[bounded_below(0.0)] for b in [bounds...]]
    prior_names = String.([propertynames(initial_parameters)...])
    prior = ParameterDistribution(prior_distns, constraints, prior_names)
    prior_mean = reshape(get_mean(prior),:)
    prior_cov = get_cov(prior)

    # We let Forward map = Loss Function evaluation
    G(u) = sqrt(nll(transform_unconstrained_to_constrained(prior, u)))
    println(nll([initial_parameters...]))

    # ℒ = ce.calibration.nll_wrapper(prior_means)
    # println("approx. scale of L in first term (data misfit) of EKI obj:", ℒ)
    # pr = norm((σ²s.^(-1/2)) .* μs)^2
    # println("approx. scale of second term (prior misfit) of EKI obj:", pr)
    # obs_noise_level = ℒ / pr
    # println("for equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about:", obs_noise_level)

    # Calibrate
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())

    for i in 1:N_iter
        params_i = get_u_final(ekiobj)
        g_ens = hcat([G(params_i[:,i]) for i in 1:N_ens]...)
        update_ensemble!(ekiobj, g_ens)
    end

    # All unconstrained
    params = mean(get_u_final(ekiobj), dims=2)
    losses = [G([mean(get_u(ekiobj, i), dims=2)...])^2 for i in 1:N_iter]
    # mean_vars = [mean(sum((get_u_at_iteration(i) .- params).^2, dims=1)) for i in 1:N_iter]
    mean_vars = [diag(cov(get_u(ekiobj, i), dims=2)) for i in 1:N_iter]

    # All unconstrained
    params = transform_unconstrained_to_constrained(prior, params)
    params = [params...] # matrix → vector

    return params, losses, mean_vars
end

function loss_reduction(nll, nll_validation, initial_parameters, kwargs)
    params, losses, mean_vars = eki(nll, initial_parameters; kwargs...)
    # println(losses[end])
    valid_loss_start = nll_validation([initial_parameters...])
    valid_loss_final = nll_validation([params...])
    println(params)
    println(losses[end] / losses[1])

    train_loss_reduction = losses[end] / losses[1]
    valid_loss_reduction = valid_loss_final / valid_loss_start
    return train_loss_reduction, valid_loss_reduction
end

## Stds within bounds

function plot_stds_within_bounds(nll, nll_validation, initial_parameters, directory; xrange=-3:0.25:5)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for stds_within_bounds = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(nll, nll_validation, initial_parameters, (stds_within_bounds = stds_within_bounds,))
        loss_reductions[stds_within_bounds] = train_loss_reduction
        val_loss_reductions[stds_within_bounds] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior std's within bounds (n_σ)", ylabel="Loss reduction (Final / Initial)", xlabel="prior std's spanned by bound width (n_σ)", legend=false, lw=3)
    plot!(loss_reductions, label="training", color=:purple, lw=4)
    plot!(val_loss_reductions, label="validation", color=:blue, lw=4)
    Plots.savefig(p, directory*"stds_within_bounds.pdf")
    println("loss-minimizing stds within bounds: $(argmin(val_loss_reductions))")
end

## Prior Variance
function plot_prior_variance(nll, nll_validation, initial_parameters, directory; xrange=0.1:0.1:1.0)
    var_loss_reductions = Dict()
    var_val_loss_reductions = Dict()
    for variance = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(nll, nll_validation, initial_parameters, (stds_within_bounds = variance,))
        var_loss_reductions[variance] = train_loss_reduction
        var_val_loss_reductions[variance] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior variance", ylabel="Loss reduction (Final / Initial)", xlabel="Prior variance")
    plot!(var_loss_reductions, label="training", lw=4, color=:purple)
    plot!(var_val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"variance.pdf")
    v = argmin(var_val_loss_reductions)
    println("loss-minimizing variance: $(v)")
    return v
end

## Number of ensemble members

function plot_num_ensemble_members(nll, nll_validation, initial_parameters, directory; xrange=1:5:30)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for N_ens = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(nll, nll_validation, initial_parameters, (N_ens = N_ens,))
        loss_reductions[N_ens] = train_loss_reduction
        val_loss_reductions[N_ens] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus N_ens", xlabel="N_ens", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
    plot!(loss_reductions, label="training")
    plot!(val_loss_reductions, label="validation")
    Plots.savefig(p, directory*"N_ens.pdf")
end

## Observation noise level

function plot_observation_noise_level(nll, nll_validation, initial_parameters, directory; xrange=-2.0:0.2:3.0)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for log_noise_level = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(nll, nll_validation, initial_parameters, (noise_level = 10.0^log_noise_level,))
        loss_reductions[log_noise_level] = train_loss_reduction
        val_loss_reductions[log_noise_level] = val_loss_reduction
    end
    p = Plots.plot(title="Loss Reduction versus Observation Noise Level", xlabel="log₁₀(Observation noise level)", ylabel="Loss reduction (Final / Initial)", legend=:topleft)
    plot!(loss_reductions, label="training", lw=4, color=:purple)
    plot!(val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"obs_noise_level.pdf")
    nl = argmin(val_loss_reductions)
    println("loss-minimizing obs noise level: $(nl)")
    return nl
end

function plot_prior_variance_and_obs_noise_level(nll, nll_validation, initial_parameters, directory; vrange=0.50:0.05:0.9, nlrange=-2.6:0.2:1.0)
    Γθs = collect(vrange)
    Γys = 10 .^ collect(nlrange)
    losses = zeros((length(Γθs), length(Γys)))
    for i in length(Γθs)
        for j in length(Γys)
            Γθ = Γθs[i]
            Γy = Γys[j]
            train_loss_reduction, val_loss_reduction = loss_reduction(nll, nll_validation, initial_parameters, (stds_within_bounds=Γθ, noise_level=Γy))
            losses[i, j] = val_loss_reduction
        end
    end
    p = Plots.heatmap(Γθs, Γys, losses, xlabel=L"\Gamma_\theta", ylabel=L"\Gamma_y", size=(200,200), yscale=:log10)
    Plots.savefig(p, directory*"GammaHeatmap.pdf")
    v = Γθs[argmin(losses)[1]]
    nl = Γys[argmin(losses)[2]]
    println("loss-minimizing Γθ: $(v)")
    println("loss-minimizing log10(Γy): $(log10(nl))")
    return v, nl
end
