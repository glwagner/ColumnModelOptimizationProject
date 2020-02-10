include("setup.jl")
include("utils.jl")

using JLD2

filename = "multi-calibration-equilibrium-10000.jld2"

@load filename results

fig, axs = subplots()

nonrotating_experiment = data.keys[2]
rotating_experiment = data.keys[5]

@show nonrotating_experiment rotating_experiment

colors = ("b", "r")
for (i, experiment) in enumerate((nonrotating_experiment, rotating_experiment))

    result = results[experiment]
    chain = result.markov_chains[end]
    visualize_markov_chain!(axs, chain, :Cᴷe, facecolor=colors[i], alpha=0.2)

end


