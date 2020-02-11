using ColumnModelOptimizationProject

@free_parameters ParametersToOptimize Cᴷu CᴷPr Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
LESdatum = LESbrary["kato, N²: 1e-5"]

# Initialize parameters with optim
nll, default_parameters = initialize_calibration(joinpath(LESbrary_path, LESdatum.filename);
                                    first_target = LESdatum.first,
                                     last_target = LESdatum.last,
                                               Δ = 0.5,
                                              Δt = 1minute,
                                   mixing_length = TKEMassFlux.EquilibriumMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux())

C_nelder_mead = optim_optimized_parameters(nll, default_parameters)

@show nll(C_nelder_mead)

# Annealing
prefix = "annealing_with_initialization"

Cᵢ = default_parameters

annealing = calibrate(nll, Cᵢ; samples=1000, iterations=3)

model = annealing.negative_log_likelihood.model
 data = annealing.negative_log_likelihood.data
 loss = annealing.negative_log_likelihood.loss
chain = annealing.markov_chains[end]

C_annealing = optimal(chain).param

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

C_nelder_mead_2 = optim_optimized_parameters(nll, C_annealing)

println("Summary:")
@printf("% 24s | ", "parameter names"); [@printf("%-12s", n) for n in propertynames(C_nelder_mead)]
@printf("\n")
@printf("% 24s | ", "Simulated annealing");  [@printf("%-12.3e", p) for p in C_annealing];
@printf("\n")
@printf("% 24s | ", "Naive Nelder-Mead");    [@printf("%-12.3e", p) for p in C_nelder_mead];
@printf("\n")
@printf("% 24s | ", "Informed Nelder-Mead"); [@printf("%-12.3e", p) for p in C_nelder_mead_2];

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], 
                                          C_nelder_mead, C_nelder_mead_2, C_annealing,
                                           fields = (:U, :T, :e), 
                                          figsize = (24, 36));
