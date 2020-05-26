using ColumnModelOptimizationProject

@free_parameters TKEParametersToOptimize Cᴷu Cᴷc Cᴷe Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

# Optimization parameters
        casename = "kato, N²: 1e-4"
         samples = 1000
              Δz = 2.0
              Δt = 1minute
relative_weights = [1e+0, 1e-4, 1e-4, 1e-6]
         LEScase = LESbrary[casename]

# Place to store results
results = @sprintf("tke_calibration_%s_dz%d_dt%d.jld2", 
                   replace(replace(casename, ", " => "_"), ": " => ""),
                   Δz, Δt/minute) 

# Run the case
annealing = calibrate_tke(joinpath(LESbrary_path, LEScase.filename), 
                                   samples = samples,
                                iterations = 3,
                                        Δz = Δz,
                                        Δt = Δt,
                              first_target = LEScase.first, 
                               last_target = LEScase.last,
                                    fields = LEScase.rotating ? (:T, :U, :V, :e) : (:T, :U, :e),
                          relative_weights = LEScase.rotating ? relative_weights : relative_weights[[1, 2, 4]],
                             mixing_length = TKEMassFlux.SimpleMixingLength(), 
                          profile_analysis = GradientProfileAnalysis(gradient_weight=0.5, value_weight=0.5))

# Save results
@save results annealing

# Do some simple analysis
model = annealing.negative_log_likelihood.model
 data = annealing.negative_log_likelihood.data
 loss = annealing.negative_log_likelihood.loss
chain = annealing.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:U, :T, :e), 
                                          figsize = (16, 6)) 

#=
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
=#
