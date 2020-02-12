using ColumnModelOptimizationProject

@free_parameters TKEParametersToOptimize Cᴷu Cᴷe Cᴰ Cᴸʷ Cʷu★ Cᴸᵇ

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

prefix = "kato_Nsq1e-6_and_Nsq1e-5"

LES_data = (LESbrary["kato, N²: 1e-6"], LESbrary["kato, N²: 1e-5"])

   cases = []
optimals = []
  models = []
les_data = []
  losses = []

for i = 1:2
    case = calibrate_tke(joinpath(LESbrary_path, LES_data[i].filename),
                               samples = 100,
                            iterations = 3,
                          first_target = LES_data[i].first,
                           last_target = LES_data[i].last,
                                     Δ = 2.0,
                                    Δt = 1minute,
                         mixing_length = TKEMassFlux.EquilibriumMixingLength(),
                        )

    # Global optimal parameters
    C★ = optimal(case.markov_chains[end]).param

    # Push the world
    push!(cases, case)
    push!(models, case.negative_log_likelihood.model)
    push!(les_data, case.negative_log_likelihood.data)
    push!(losses, case.negative_log_likelihood.loss)
    push!(optimals, C★)
end

@save prefix * "_two_cases.jld2" cases

#
# With own and other's optimal parameters.
#

# First, the results:
fig, axs= visualize_realizations(models[1], les_data[1], losses[1].targets[[1, end]], optimals[1],
                                 fields=(:U, :T, :e), figsize = (16, 6))

fig.suptitle("Non rotating case with its own optimal paramteters")

fig, axs= visualize_realizations(models[2], les_data[2], losses[2].targets[[1, end]], optimals[2],
                                 fields=(:U, :T, :e), figsize = (16, 6))

fig.suptitle("Rotating case with its own optimal paramteters")


# Now, with each other's optimal parameters
fig, axs= visualize_realizations(models[2], les_data[2], losses[2].targets[[1, end]], optimals[1],
                                 fields=(:U, :T, :e), figsize = (16, 6))

fig.suptitle("Rotating, with optimal paramteters for the non-rotating case.")

# Vice versa
fig, axs= visualize_realizations(models[1], les_data[1], losses[1].targets[[1, end]], optimals[2],
                                 fields=(:U, :T, :e), figsize = (16, 6))

fig.suptitle("Non-rotating, with optimal paramteters for the rotating case.")


fig, axs = subplots()

plot(optimals[1], "o", 
     label="Optimal parameters, non-rotating, \$ N^2 = 10^{-6} \\, \\mathrm{s^{-2}} \$")

plot(optimals[2], "o", 
     label="Optimal parameters, non-rotating, \$ N^2 = 10^{-4} \\, \\mathrm{s^{-2}} \$")
