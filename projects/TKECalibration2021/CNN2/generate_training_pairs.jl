"""
Run this script once to generate training pairs and save them to `training_pairs.jld2`.
Most of the relevant functions are contained in `./generate_training_pairs_utils.jld2`.
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
@free_parameters(CATKEparameters, Cᴷu⁻, Cᴷu⁺, Cᴷc⁻, Cᴷc⁺, Cᴷe⁻, Cᴷe⁺)

include("generate_training_pairs_utils.jl")

function generate_training_pairs(training_simulations; Nt = 6,
                                                       Δt = 10.0,
                                                       profile_indices = profile_indices,
                                                       data_indices = data_indices,
                                                      )
    training_pairs = []

    defaults = TKEParametersRiDependent([0.7213, 0.7588, 0.1513, 0.7337, 0.3977, 1.7738, 0.1334, 1.2247, 2.9079, 1.1612, 3.618765767212402, 1.3051900286568907])
    # defaults = TKEParametersConvectiveAdjustmentRiIndependent([[0.1723  0.0676  0.6067  2.6969  2.3674  0.1204  2.6416  0.1023  1.3936  0.3576]...])
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
                                              eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities(),
                                                    tke_equation = TKEMassFlux.TKEParameters(),
                                           # convective_adjustment = TKEMassFlux.VariablePrandtlConvectiveAdjustment(),
                                             )

         # Set model to custom defaults
         set!(model, defaults)

         initial_parameters = custom_defaults(model, CATKEparameters)

         # plot_prior_variance_and_obs_noise_level(CATKEparameters, model, data, fields, weights, directory)
         end_index = length(data)-Nt
         @showprogress for start_index = 13:end_index
         # @showprogress for start_index = 200:202
             # @info "Generating pairs for time $(start_index)/$(end_index) ($(round(start_index*100/end_index))%)"
             # parameters = get_optimal_αs_σs_eki(CATKEparameters, model, data, fields, weights, start_index; Nt=6,
             #                                            noise_level = 10^(-2.0),
             #                                            prior_variance = 0.6,
             #                                            N_ens=20,
             #                                            N_iter=20)

             parameters = get_optimal_αs_σs_lbfgs(CATKEparameters, initial_parameters, model, data, fields, weights, start_index; Nt=6, time_limit=0.25)

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
@save "CNN/training_pairs2.jld2" xtrain ytrain

xtest, ytest = generate_training_pairs(TwoDaySuite)
@save "CNN/testing_pairs2.jld2" xtest ytest

# training_simulations = Dict("4d_strong_wind_weak_cooling" => FourDaySuite["4d_strong_wind_weak_cooling"])
# x_lims = (minimum(xtrain), maximum(xtrain))
# Nz = length(profile_indices)
# labels=["U", "V", "T", "E"]
# anim = @animate for i = 1:length(xtrain[:,1])
#     p = Plots.plot(grid=false, size=(180,500), framestyle=:none, xlims = x_lims, foreground_color_legend = nothing, title="$(round(i*10/144)/10) days")
#     for fieldindex = 1:4
#         start = (fieldindex-1)*Nz + 1
#         zindices = start:start+Nz-1
#         Plots.plot!(xtrain[zindices, i], zindices, lw=4, legend=:topleft, label=labels[fieldindex], legendfontsize=12)
#     end
#     p
# end
# gif(anim, "CNN_input.gif")

# function plot_prior_variance_and_obs_noise_level(CATKEparameters, model, data, fields, weights, directory; vrange=0.40:0.10:0.90, nlrange=-20.0:1.0:-2.0)
# # function plot_prior_variance_and_obs_noise_level(CATKEparameters, model, data, fields, weights, directory; vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
#     Γθs = collect(vrange)
#     Γys = 10 .^ collect(nlrange)
#     losses = zeros((length(Γθs), length(Γys)))
#     counter = 1
#     countermax = length(Γθs)*length(Γys)
#     for i in 1:length(Γθs)
#         for j in 1:length(Γys)
#             println("progress $(counter)/$(countermax)")
#             Γθ = Γθs[i]
#             Γy = Γys[j]
#             params = get_optimal_αs_σs(CATKEparameters, model, data, fields, weights, 200;
#                                                Nt=Nt,
#                                                noise_level = Γy,
#                                                prior_variance = Γθ,
#                                                N_ens=20,
#                                                N_iter=20
#                                        )
#             losses[i, j] = evolve_from_start_index(CATKEparameters(params), model, data, fields, weights, 200; Nt=Nt)
#             counter += 1
#         end
#     end
#     p = Plots.heatmap(Γθs, Γys, losses, xlabel=L"\Gamma_\theta", ylabel=L"\Gamma_y", size=(250,250), yscale=:log10, clims=(0.0,10.0))
#     Plots.savefig(p, directory*"GammaHeatmap.pdf")
#     v = Γθs[argmin(losses)[1]]
#     nl = Γys[argmin(losses)[2]]
#     println("loss-minimizing Γθ: $(v)")
#     println("loss-minimizing log10(Γy): $(log10(nl))")
#     return v, nl
# end
#
# """
# Calibrate TKEParametersConvectiveAdjustmentRiIndependent
# Store `training_pairs` to JSON file.
# """
#
# Truth = Dict()
# CATKE = Dict()
# for mynll in ce.validation.nll.batch
#     Truth[mynll.data.name] = mynll.data
#     CATKE[mynll.data.name] = full_time_series(parameters, mynll)
# end
#
# include("../visualize.jl")
# whole_suite_animation(500)
# anim = @animate for t=1:2:840
#     p = whole_suite_animation(t)
# end
