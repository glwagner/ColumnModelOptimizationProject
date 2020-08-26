using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

# Optimization parameters
        casename = "convection, N²: 2e-6"
         samples = 1000
              Δz = 2.0
              Δt = 1minute
relative_weights = [1e+0, 1e-4, 1e-4, 1e-4]
         LEScase = LESbrary[casename]

# Place to store results
results = @sprintf("tke_calibration_%s_dz%d_dt%d.jld2", 
                   replace(replace(casename, ", " => "_"), ": " => ""),
                   Δz, Δt/minute) 

nll, _ = init_tke_calibration(LEScase.filename;
                                              Δz = Δz,
                                              Δt = Δt,
                                    first_target = LEScase.first, 
                                     last_target = LEScase.last,
                                          fields = tke_fields(LEScase),
                                relative_weights = tke_relative_weights(LEScase),
                              eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities(),
                                   mixing_length = TKEMassFlux.SimpleMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                                    tke_equation = TKEMassFlux.TKEParameters(),
                                      parameters = RiDependentTKEParameters,
                              )

initial_parameters = RiDependentTKEParameters(
                                              Cᴷu⁰ = 0.2,
                                              Cᴷuᵟ = 0.1,
                                              Cᴷuʷ = 1.2,
                                              Cᴷuᶜ = -0.1,
                                              Cᴷc⁰ = 1.0,
                                              Cᴷcᵟ = -0.8,
                                              Cᴷcʷ = 0.1,
                                              Cᴷcᶜ = 0.1,
                                              Cᴷe⁰ = 2.0,
                                              Cᴷeᵟ = -0.5,
                                              Cᴷeʷ = 0.5,
                                              Cᴷeᶜ = 0.1,
                                              Cᴰ   = 3.0,
                                              Cᴸʷ  = 1.2,
                                              Cᴸᵇ  = 1.5,
                                              Cʷu★ = 2.2,
                                              CʷwΔ = 1.1,
                                             )

# Run the case
calibration = calibrate(nll, initial_parameters, samples = 1000, iterations = 4)

# Save results
@save results calibration

# Do some simple analysis
model = calibration.negative_log_likelihood.model
 data = calibration.negative_log_likelihood.data
 loss = calibration.negative_log_likelihood.loss
chain = calibration.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           #fields = (:U, :V, :T, :e), 
                                           fields = (:T, :e), 
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
