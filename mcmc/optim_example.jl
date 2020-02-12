using ColumnModelOptimizationProject, JLD2

@free_parameters ParametersToOptimize Cᴷu Cᴷe CᴷPr Cᴰ Cᴸʷ Cʷu★ Cᴸᵇ

lower_bounds = [0.0 for p in propertynames(ParametersToOptimize)]
upper_bounds = [10.0 for p in propertynames(ParametersToOptimize)]

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

prefix = "kato"

LESdata = (
           LESbrary["kato, N²: 1e-7"],
           LESbrary["kato, N²: 2e-7"],
           LESbrary["kato, N²: 5e-7"],
           LESbrary["kato, N²: 1e-6"],
           LESbrary["kato, N²: 1e-5"],
           LESbrary["kato, N²: 1e-4"],
           )

residuals = []

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
LESdatum = LESdata[4]

nll, default_parameters = initialize_calibration(joinpath(LESbrary_path, LESdatum.filename);
                                    first_target = LESdatum.first,
                                     last_target = LESdatum.last,
                                               Δ = 4.0,
                                              Δt = 1minute,
                                   mixing_length = TKEMassFlux.EquilibriumMixingLength(),
                                   #mixing_length = TKEMassFlux.SimpleMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux())

optim_nll = OptimSafeNegativeLogLikelihood(default_parameters, nll)

println("Optimizing to $(LESdatum.filename) using the Nelder-Mead algorithm.")

@time residual = optimize(optim_nll, Array(default_parameters))

model, data, loss = nll.model, nll.data, nll.loss
C★ = typeof(default_parameters)(minimizer(residual))

viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:U, :T, :e), 
                                          figsize = (12, 4)) 


