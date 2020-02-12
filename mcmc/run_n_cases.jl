using ColumnModelOptimizationProject, JLD2, Printf

@free_parameters ParametersToOptimize Cᴷu Cᴷe CᴷPr Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

prefix = "kato"

LES_data = (
            #LESbrary["kato, N²: 1e-7"],
            #LESbrary["kato, N²: 2e-7"],
            #LESbrary["kato, N²: 5e-7"],
            LESbrary["kato, N²: 1e-6"],
            LESbrary["kato, N²: 1e-5"],
            LESbrary["kato, N²: 1e-4"],
           )

   cases = []
optimals = []
  models = []
  datums = []
  losses = []

for i = 1:length(LES_data)

    println("""

    Calibrating a TKE, mass-flux model from LES data at 
    
    $(LES_data[i].filename)

    Please stand by.
    """)

    case = calibrate(joinpath(LESbrary_path, LES_data[i].filename),
                             samples = 2000,
                          iterations = 5,
                        first_target = LES_data[i].first,
                         last_target = LES_data[i].last,
                                   Δ = 4.0,
                                  Δt = 1minute,
                       mixing_length = TKEMassFlux.EquilibriumMixingLength(),
                       #mixing_length = TKEMassFlux.SimpleMixingLength(),
                      tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                      )

    println("Calibration to $LESbrary_path complete.")

    # Global optimal parameters
    C★ = optimal(case.markov_chains[end]).param

    # Push the world
    push!(cases, case)
    push!(models, case.negative_log_likelihood.model)
    push!(datums, case.negative_log_likelihood.data)
    push!(losses, case.negative_log_likelihood.loss)
    push!(optimals, C★)
end

@save prefix * "_low_strat.jld2" cases

symbols = ["o", "^", "s", "*", "d", "<"]

fig, axs = subplots()

for (i, opt) in enumerate(optimals)
    N² = buoyancy_frequency(datums[i])
    plot(opt, symbols[i], label=@sprintf("\$ N^2 = %.1e \\, \\mathrm{s^{-2}} \$", N²), alpha=0.6)
end

tick_labels = [ parameter_latex_guide[p] for p in propertynames(optimals[1]) ]
xticks(0:length(tick_labels)-1, tick_labels)

legend()

axs.set_yscale("log")


