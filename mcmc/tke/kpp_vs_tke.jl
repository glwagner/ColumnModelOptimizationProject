using ColumnModelOptimizationProject

@free_parameters ParametersToOptimize Cᴷu CᴷPr Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
#LESdatum = LESbrary["kato, N²: 1e-5"]
LESdatum = LESbrary["ekman, N²: 1e-5"]

Δ = 2.0
Δt = 1minute

tke_nll, tke_defaults = initialize_calibration(joinpath(LESbrary_path, LESdatum.filename);
                                                    first_target = LESdatum.first,
                                                     last_target = LESdatum.last,
                                                         fields = (:T, :U, :V, :e),
                                               relative_weights = [1e4, 1e2, 1e2, 1],
                                               #           fields = (:T, :U, :e),
                                               # relative_weights = [1e4, 1e2, 1],
                                                               Δ = Δ,
                                                              Δt = Δt,
                                                   mixing_length = TKEMassFlux.EquilibriumMixingLength(),
                                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux())

kpp_nll, kpp_defaults = initialize_KPP_calibration(joinpath(LESbrary_path, LESdatum.filename);
                                                        first_target = LESdatum.first,
                                                         last_target = LESdatum.last,
                                                             fields = (:T, :U, :V),
                                                   relative_weights = [1e2, 1, 1],
                                                   #           fields = (:T, :U),
                                                   # relative_weights = [1e2, 1],
                                                                   Δ = Δ,
                                                                  Δt = Δt)

tke_results = calibrate(tke_nll, tke_defaults; samples=1000, iterations=3)
kpp_results = calibrate(kpp_nll, kpp_defaults; samples=1000, iterations=3)


data = tke_results.negative_log_likelihood.data

tke_model = tke_results.negative_log_likelihood.model
kpp_model = kpp_results.negative_log_likelihood.model

tke_loss = tke_results.negative_log_likelihood.loss
kpp_loss = kpp_results.negative_log_likelihood.loss

tke_chain = tke_results.markov_chains[end]
kpp_chain = kpp_results.markov_chains[end]

tke_C★ = optimal(tke_chain).param
kpp_C★ = optimal(kpp_chain).param

close("all")

tke_fig, tke_axs = visualize_realizations(tke_model, data, tke_loss.targets[[1, end]], tke_C★,
                                           fields = (:U, :V, :T, :e), 
                                          figsize = (12, 4))

kpp_fig, kpp_axs = visualize_realizations(kpp_model, data, kpp_loss.targets[[1, end]], kpp_C★,
                                           fields = (:U, :V, :T), 
                                          figsize = (12, 4))
