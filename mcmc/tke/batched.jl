using ColumnModelOptimizationProject, JLD2

@free_parameters ParametersToOptimize Cᴷu CᴷPr Cᴰ Cᴸʷ Cʷu★ Cᴸᵇ

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LESdata = (
            LESbrary["kato, N²: 1e-7"], 
            LESbrary["kato, N²: 2e-7"], 
            LESbrary["kato, N²: 5e-7"], 
            LESbrary["kato, N²: 1e-6"], 
            LESbrary["kato, N²: 1e-5"], 
            LESbrary["ekman, N²: 1e-6"],
            LESbrary["ekman, N²: 1e-5"],
            LESbrary["ekman, N²: 1e-4"]
           )

# Generating function for kwargs
function calibration_kwargs(datum)
              fields = datum.rotating ? ( :T,   :U,   :V,   :e) : ( :T,   :U,   :e)
    relative_weights = datum.rotating ? [1.0, 1e-2, 1e-2, 1e-4] : [1.0, 1e-2, 1e-4]
    return (
                first_target = datum.first,
                 last_target = datum.last,
                      fields = fields,
            relative_weights = relative_weights,
               mixing_length = TKEMassFlux.EquilibriumMixingLength(),
              tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
           )
end

# Create batch
batch = []

for datum in LESdata
    nll, default_parametetrs = initialize_TKEMassFlux_calibration(datum.filename; Δ=1.0, Δt=30second, 
                                                          calibration_kwargs(datum)...)
    push!(batch, nll)
end

batched_nll = BatchedNegativeLogLikelihood([nll for nll in batch])
default_parameters = DefaultFreeParameters(batch[1].model, ParametersToOptimize)

anneling = calibrate(batched_nll, default_parameters, samples=10000, iterations=3);


