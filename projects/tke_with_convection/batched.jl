using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LESdata = (
           LESbrary["kato, N²: 1e-7"],
           LESbrary["kato, N²: 1e-6"],
           LESbrary["kato, N²: 1e-5"],
           LESbrary["kato, N²: 1e-4"],
           LESbrary["ekman, N²: 1e-7"],
           LESbrary["ekman, N²: 1e-6"],
           LESbrary["ekman, N²: 1e-5"],
           LESbrary["ekman, N²: 1e-4"],
           LESbrary["convection, N²: 2e-6"],
           LESbrary["convection, N²: 1e-5"],
          )

# Generating function for kwargs
function calibration_kwargs(datum)

    fields = tke_fields(datum)
    relative_weights = tke_relative_weights(datum)

    return (
                first_target = datum.first,
                 last_target = datum.last,
                      fields = fields,
            relative_weights = relative_weights,
               mixing_length = TKEMassFlux.SimpleMixingLength(),
              tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
           )
end

# Create batch
batch = []
models = []
data = []

for datum in LESdata
    nll, default_parametetrs = init_tke_calibration(datum.filename;
                                                     Δ = 2.0,
                                                    Δt = 1minute,
                                                    calibration_kwargs(datum)...)
    push!(batch, nll)
    push!(models, nll.model)
    push!(data, nll.data)
end

batched_nll = BatchedNegativeLogLikelihood([nll for nll in batch])
default_parameters = DefaultFreeParameters(batch[1].model, TKEParametersToOptimize)

annealing = calibrate(batched_nll, default_parameters, samples=10000, iterations=10);

println("done")
