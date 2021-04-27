## Optimizing TKE parameters
using TKECalibration2021
using Dao
using Distributions
using LinearAlgebra
using Random
using Plots

LESdata = TwoDaySuite # Calibration set
LESdata_validation = FourDaySuite # Validation set
RelevantParameters = TKEParametersRiIndependent
ParametersToOptimize = TKEParametersRiIndependent

# define closure here cause ParametersToOptimize has to be in the global scope
function loss_closure(nll)
        ℒ(parameters::ParametersToOptimize) = nll(parameters)
        ℒ(parameters::Vector) = nll(ParametersToOptimize([parameters...]))
        return ℒ
end

nll, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize; loss_closure = loss_closure, relative_weights = relative_weights)
initial_parameters = ParametersToOptimize([0.1320799067908237, 0.21748565946199314, 0.051363488558909924, 0.5477193236638974, 0.8559038503413254, 3.681157252463703, 2.4855193201082426])

nll(initial_parameters)
# ℱ = model_time_series(default_parameters, model, cdata, loss_function)
# myloss(ℱ) = loss_function(ℱ, cdata)
# myloss(ℱ)

##

"""
nll
initial_parameters

Keyword Arguments
set_prior_means_to_initial_parameters:
n_obs: Number of synthetic observations from G(u)
noise_level: Observation noise level
N_ens: number of ensemble members, J
N_iter: number of EKI iterations, Ns
stds_within_bounds:
"""
function ensemble_kalman_inversion(nll, initial_parameters;
                                                    set_prior_means_to_initial_parameters = true,
                                                    n_obs = 1,
                                                    noise_level = 1e-7,
                                                    N_ens = 100,
                                                    N_iter = 10,
                                                    stds_within_bounds = 5
                                                    )

    bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = set_prior_means_to_initial_parameters ? [initial_parameters...] : mean.(bounds)

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
        g_ens = hcat([G(params_i.stored_data[:,i]) for i in 1:N_ens]...)
        update_ensemble!(ekiobj, g_ens)
    end

    losses = [G([mean(ekiobj.u[i].stored_data, dims=2)...]) for i in 1:N_iter]

    A(i) = ekiobj.u[i].stored_data
    params = mean(ekiobj.u[end].stored_data, dims=2)
    mean_vars = [mean(sum((A(i) .- params).^2, dims=1)) for i in 1:N_iter]

    params = get_u(ekiobj)

    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{Real})

    return params, losses, mean_vars
end

function loss_reduction(kwargs)
    params, losses, mean_vars = ensemble_kalman_inversion(nll, initial_parameters; kwargs...)
    # println(losses[end])
    println(params)
    println(losses[end] / losses[1])
    return losses[end] / losses[1]
end

## Distance from final parameter mean
params, losses, mean_vars = ensemble_kalman_inversion(nll, initial_parameters; N_iter=50, set_prior_means_to_initial_parameters=false, stds_within_bounds=5)
p = Plots.plot(1:length(mean_vars), mean_vars, yscale=:log10, title="Avg. particle distance from final ensemble mean", xlabel="Iteration", ylabel="Distance (parameter space)", legend=false, lw=3)
Plots.savefig(p, "prior_mean_center_bound____distance_from_final_mean_N_iter_50.pdf")


## Stds within bounds

# loss_reduction((stds_within_bounds = 3, set_prior_means_to_initial_parameters=false))
loss_reductions = Dict()
for stds_within_bounds = 3:0.5:10
    loss_reductions[stds_within_bounds] = loss_reduction((stds_within_bounds = stds_within_bounds, set_prior_means_to_initial_parameters=false))
end
p = Plots.plot(loss_reductions, title="Loss reduction versus prior std's within bounds (n_σ)", ylabel="Loss reduction (Final / Initial)", xlabel="prior std's spanned by bound width (n_σ)", legend=false, lw=3)
Plots.savefig(p, "prior_mean_center_bound____stds_within_bounds.pdf")

## Number of ensemble members

loss_reductions = []
for N_ens = 25:25:500
    println(N_ens)
    push!(loss_reductions, loss_reduction((N_ens = N_ens, set_prior_means_to_initial_parameters=false)))
end
p = Plots.plot(25:25:500, loss_reductions, title="Loss reduction versus N_ens", xlabel="N_ens", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, "prior_mean_center_bound____N_ens2.pdf")

## Observation noise level

loss_reductions = []
for log_noise_level = -10:0.5:-1
    push!(loss_reductions, loss_reduction((N_iter=5, noise_level = 10.0^log_noise_level, set_prior_means_to_initial_parameters=false)))
end
p = Plots.plot(-10:0.5:-1, loss_reductions, title="Loss reduction versus observation noise level", xlabel="log₁₀(Observation noise level)", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, "prior_mean_center_bound____obs_noise_level.pdf")
