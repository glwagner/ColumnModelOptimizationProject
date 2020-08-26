include("setup.jl")
include("utils.jl")

using JLD2

results = OrderedDict()

nresults = 7
samples = 10000
prefix = "equilibrium"

for i = 1:nresults
    result = calibrate(data.keys[i], samples=samples, first_target=data.vals[i].first, last_target=data.vals[i].last,
                       mixing_length = TKEMassFlux.EquilibriumMixingLength(), Δ=1.0)

    results[data.keys[i]] = result

    model = result.negative_log_likelihood.model
    column_data = result.negative_log_likelihood.data
    loss = result.negative_log_likelihood.loss
    chain = result.markov_chains[end]

    viz_fig, viz_axs = visualize_realizations(model, column_data, loss.targets[[1, end]], 
                                              optimal(chain).param, fields=(:U, :T, :e), 
                                              figsize=(24, 36)) 

    last_target = loss.targets[end]
    name = data.keys[i][1:end-5]
    plotname = replace(name, "_" => " ")
    viz_fig.suptitle("$prefix, $plotname, last = $last_target")
    savefig(prefix * "_" * name * ".png", dpi=480)
end

@save "multi-calibration-equilibrium-$samples.jld2" results

fig, axs = subplots(ncols=4, nrows=2)

chain = results.vals[1].markov_chains[end]
parameters = chain[1].param

for (j, p) in enumerate(propertynames(parameters))
    axs[j].set_xscale("log")

    sca(axs[j])
    xlabel(L"N^2 \, \mathrm{(s^{-2})}")
    ylabel(parameter_latex_guide[p])
    grid()
end


for (i, datapath) in enumerate(keys(results))
    N² = data[datapath].N²
    color = data[datapath].rotating ? "b" : "r"
    note = data[datapath].rotating ? "rotating" : "non-rotating"

    chain = results[datapath].markov_chains[end]

    optimal_parameters = optimal(chain).param
    median_parameters = median(collect_samples(chain), dims=2)
    mean_parameters = mean(collect_samples(chain), dims=2)

    for j = 1:length(parameters)
        sca(axs[j])

        if j == 1 && i == 1
            plot(N², optimal_parameters[j], linestyle="None", marker="*", markersize=10,             color=color, alpha=0.4, label="optimal, $note")
            plot(N², median_parameters[j],  linestyle="None", marker="o", markersize=4,  mfc="none", color=color, alpha=0.4, label="median, $note")
            plot(N², mean_parameters[j],    linestyle="None", marker="^", markersize=4,  mfc="none", color=color, alpha=0.4, label="mean, $note")
        else
            plot(N², optimal_parameters[j], linestyle="None", marker="*", markersize=10,             color=color, alpha=0.4)
            plot(N², median_parameters[j],  linestyle="None", marker="o", markersize=4,  mfc="none", color=color, alpha=0.4)
            plot(N², mean_parameters[j],    linestyle="None", marker="^", markersize=4,  mfc="none", color=color, alpha=0.4)
        end
    end
end

tight_layout()
sca(axs[1])
legend(fontsize=6, markerscale=0.5)

savefig("calibration_summary.png", dpi=480)
