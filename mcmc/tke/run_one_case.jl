include("setup.jl")
include("utils.jl")

using JLD2

results = OrderedDict()

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LES_data = LESbrary["kato, N²: 1e-5"]

samples = 1000
prefix = "test"

annealing = calibrate(joinpath(LESbrary_path, LES_data.filename), 
                         samples = samples, 
                      iterations = 100,
                    first_target = LES_data.first, 
                     last_target = LES_data.last,
                   mixing_length = TKEMassFlux.EquilibriumMixingLength(), Δ=1.0)

model = annealing.negative_log_likelihood.model
 data = annealing.negative_log_likelihood.data
 loss = annealing.negative_log_likelihood.loss
chain = annealing.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:U, :T, :e), 
                                          figsize = (24, 36)) 

fig, axs = subplots(ncols=2, figsize=(16, 6))

optimums = optimum_series(annealing)
errors = [optimal(chain).error for chain in annealing.markov_chains]

for (i, name) in enumerate(propertynames(optimums))
    series = optimums[i]
    final_value = series[end]
    lbl = parameter_latex_guide[name]

    sca(axs[1])
    plot(series / final_value, linestyle="-", marker="o", markersize=5, linewidth=1, label=lbl)
end

legend()

sca(axs[2])
plot(errors / errors[1], linestyle="-", marker="o", markersize=5, linewidth=1)

