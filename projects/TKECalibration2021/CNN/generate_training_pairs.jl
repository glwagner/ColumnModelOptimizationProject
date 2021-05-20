"""
Run this script once to generate training pairs and save them to `training_pairs.jld2`.
"""

using TKECalibration2021
using ColumnModelOptimizationProject
using ColumnModelOptimizationProject: TKEMassFluxOptimization, TKEMassFlux,
                                      run_until!, mean_variance, nan2inf, set!
using ColumnModelOptimizationProject.TKEMassFluxOptimization: set!

using OceanTurb: CellField
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using Distributions, Random, LinearAlgebra
using ProgressMeter: @showprogress
using Optim

# Saving to output file
using JLD2

"""
For each training simulation
    For each time step i ∈ 1:Nt-60 in the simulation
        run line-search gradient descent on α_ϕ, α_ϕ
            where the loss at each iteration of the descent is computed by
            (1) initializing the CATKE model to the LES profiles at that time step (ϕ_CATKE[0] ← ϕ_LES[i])
            (2) evolving CATKE forward for 6 time steps (corresponding to 1 hour in simulation time because Δt=10 mins)
            (3) evaluating (1/64)||ϕ_CATKE[6] - ϕ_LES[i+6]||^2, the MSE between the CATKE prediction for ϕ and the LES solution for ϕ.
"""

# p =  [αu, αt, αe, βu, βt, βe]
@free_parameters(CATKEparameters,
                  Cᴬu, Cᴬc, Cᴬe,
                  Cᴷu, Cᴷc, Cᴷe)

using Statistics: mean

profile_indices = 25:64
data_indices = 50:127

function get_normalization_functions(LESdata)
  normalize_function = Dict()
  for field in (:U, :V, :T, :e)

      μs = []
      σs = []
      for LEScase in values(LESdata)
          data = ColumnData(LEScase.filename)
          fields = !(LEScase.stressed) ? (:T, :e) :
                   !(LEScase.rotating) ? (:T, :U, :e) :
                                         (:T, :U, :V, :e)

          # first = LEScase.first
          # last = LEScase.last == nothing ? length(data) : LEScase.last
          # targets = (first, last)

          if field in fields
              push!(μs, profile_mean(data, field; indices=data_indices))
              # push!(σs, sqrt(max_variance(data, field, targets)))
              push!(σs, mean_std(data, field; indices=data_indices))

          end
      end
      μ = mean(μs)
      σ = mean(σs)
      normalize(Φ) = (Φ .- μ) ./ σ
      μ = 0.0
      if field == :T; μ = mean(μs); end
      normalize_function[field] = normalize
  end

  return normalize_function
end

normalize_function = get_normalization_functions(FourDaySuite)

function get_UVTE(U, V, T, E)
    U = normalize_function[:U](U)[profile_indices]
    V = normalize_function[:V](V)[profile_indices]
    T = normalize_function[:T](T)[profile_indices]
    E = normalize_function[:e](E)[profile_indices]

    UVTE = vcat(reverse(U), V, T, E)
    return UVTE
end

function lognormal_μ_σ²(mean, variance)
    k = variance / mean^2 + 1
    μ = log(mean / sqrt(k))
    σ² = log(k)
    return μ, σ²
end

"""
Using L-BFGS to minimize the loss on `fields`.
"""
function get_optimal_αs_σs_lbfgs(CATKEparameters, initial_parameters, model, data, fields, weights, start_index; Nt=6, time_limit = 3.0)

    # params =  [αu, αt, αe, σu, σt, σe]
    loss(params) = evolve_from_start_index(CATKEparameters(params), model, data, fields, weights, start_index; Nt=Nt)

    result = try
        Optim.optimize(loss, [initial_parameters...], BFGS(), Optim.Options(time_limit = time_limit))
    catch
        nothing
    end

    if result == nothing
        @warn "Returned infinite loss. Setting output parameters to defaults."
        return initial_parameters
    end

    params = Optim.minimizer(result)

    if start_index == 1
        println(result)
        println("Loss on initial parameters: $(loss(initial_parameters))")
    end

    return params
end


"""
Use Ensemble Kalman Inversion to minimize the loss on `fields`.
"""
function get_optimal_αs_σs_eki(CATKEparameters, initial_parameters, model, data, fields, weights, start_index; Nt=6,
                                            noise_level = 10^(-2.0),
                                            prior_variance = 0.5,
                                            N_ens=20,
                                            N_iter=20)

    # params =  [αu, αt, αe, σu, σt, σe]
    loss(params) = evolve_from_start_index(CATKEparameters(params), model, data, fields, weights, start_index; Nt=Nt)

    prior_variances = [prior_variance for i in initial_parameters]
    prior_means = [initial_parameters...]
    prior_distns = [Parameterized(Normal([lognormal_μ_σ²(prior_means[i], prior_variances[i], bounds[i])...]...)) for i in 1:length(bounds)]

    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Independent noise for synthetic observations
    Γy = noise_level * Matrix(I, 1, 1)

    # Loss Function Minimum
    y_obs = [0.0]

    # Define Prior
    constraints = [[bounded_below(0.0)] for b in 1:6]
    prior_names = String.([propertynames(CATKEparameters([0.0 for i=1:6]))...])
    prior = ParameterDistribution(prior_distns, constraints, prior_names)
    prior_mean = reshape(get_mean(prior),:)
    prior_cov = get_cov(prior)

    # We let Forward map = Loss Function evaluation
    G(u) = sqrt(loss(transform_unconstrained_to_constrained(prior, u)))

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

    # All unconstrained
    params = transform_unconstrained_to_constrained(prior, params)
    params = [params...] # matrix → vector

    return params

end

function evolve_from_start_index(parameters, column_model, column_data, fields, weights, start_index; Nt=6, return_UVTE=false)

    initialize_forward_run!(column_model, column_data, parameters, start_index)

    grid = column_model.grid

    output = ModelTimeSeries([CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt])

    U = column_model.solution.U
    V = column_model.solution.V
    T = column_model.solution.T
    e = column_model.solution.e

    if return_UVTE
        U = [U.data...]
        V = [V.data...]
        T = [T.data...]
        e = [e.data...]
        return U, V, T, e
    end

    total_discrepancy = zero(eltype(column_model.grid))
    coarse_data = discrepancy = CellField(grid)

    for i in 1:Nt

        # Simulation time step
        target = i + start_index - 1

        # Evolve model for Nt timesteps
        run_until!(column_model.model, column_model.Δt, column_data.t[target])

        # Set snapshot to empty CellField
        U_snapshot = output.U[i].data
        V_snapshot = output.V[i].data
        T_snapshot = output.T[i].data
        e_snapshot = output.e[i].data

        # Fill empty CellField with model data
        U_snapshot .= U.data
        V_snapshot .= V.data
        T_snapshot .= T.data
        e_snapshot .= e.data

        for (field_index, field_name) in enumerate(fields)

            model_field = getproperty(column_model.solution, field_name)
            data_field = getproperty(column_data, field_name)[target]

            set!(coarse_data, data_field)

            for i in eachindex(discrepancy)
                @inbounds discrepancy[i] = (coarse_data[i] - model_field[i])^2
            end

            total_discrepancy += weights[field_index] * mean(discrepancy)
        end
    end

    return nan2inf(total_discrepancy)
end

function estimate_weights(data, relative_weights, field_names)
    mean_variances = [mean_variance(data, field) for field in field_names]
    weights = [1/σ for σ in mean_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    return weights
end

function generate_training_pairs(training_simulations; Nt = 6,
                                                       Δt = 10.0,
                                                       profile_indices = profile_indices,
                                                       data_indices = data_indices,
                                                      )
    training_pairs = []

    defaults = TKEParametersConvectiveAdjustmentRiIndependent([[0.1723  0.0676  0.6067  2.6969  2.3674  0.1204  2.6416  0.1023  1.3936  0.3576]...])
    default_count = 0

    for LEScase in values(training_simulations)
        fields = !(LEScase.stressed) ? (:T, :e) :
                 !(LEScase.rotating) ? (:T, :U, :e) :
                                       (:T, :U, :V, :e)

        data = ColumnData(LEScase.filename)

        @info "Generating pairs for LEScase $(data.name)"

        relative_weights = [1.0 for field in fields]

        weights = estimate_weights(data, relative_weights, fields)

        model = TKEMassFluxOptimization.ColumnModel(data, Δt,
                                                              N = 64,
                                                   mixing_length = TKEMassFlux.SimpleMixingLength(),
                                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                                              eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                                                    tke_equation = TKEMassFlux.TKEParameters(),
                                           convective_adjustment = TKEMassFlux.VariablePrandtlConvectiveAdjustment(),
                                             )

         # Set model to custom defaults
         set!(model, defaults)

         initial_parameters = CATKEparameters([0.1023, 1.3936, 0.3576, 0.1723, 0.0676, 0.6067])

         end_index = length(data)-Nt
         @showprogress for start_index = 1:end_index

             parameters = get_optimal_αs_σs_lbfgs(CATKEparameters, initial_parameters, model, data, fields, weights, start_index; Nt=6, time_limit=5.0)

             if parameters == initial_parameters
                default_count += 1
             end

             U, V, T, E = evolve_from_start_index(CATKEparameters(parameters), model, data, fields, weights, start_index; Nt=6, return_UVTE=true)

             UVTE = get_UVTE(U, V, T, E)

             training_pair = (UVTE, parameters)

             push!(training_pairs, training_pair)
         end
    end
    xtrain = hcat([pair[1] for pair in training_pairs]...)
    ytrain = hcat([pair[2] for pair in training_pairs]...)

    println("Default count: $default_count")
    return xtrain, ytrain
end

xtrain, ytrain = generate_training_pairs(FourDaySuite)
@save "CNN/training_pairs.jld2" xtrain ytrain

xtest, ytest = generate_training_pairs(TwoDaySuite)
@save "CNN/testing_pairs.jld2" xtest ytest
