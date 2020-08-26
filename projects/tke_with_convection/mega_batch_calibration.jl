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

weights = [1.0 for d in LESdata]
           
# Generating function for kwargs
function calibration_kwargs(datum)

    fields = tke_fields(datum)
    relative_weights = tke_relative_weights(datum)

    return (
                  first_target = datum.first,
                   last_target = datum.last,
                        fields = fields,
              relative_weights = relative_weights,
            eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities(),
                 mixing_length = TKEMassFlux.SimpleMixingLength(),
                tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                  tke_equation = TKEMassFlux.TKEParameters(),
                    parameters = RiDependentTKEParameters,
           )
end

# Create batch
batch = []
models = []
data = []

for datum in LESdata
    nll, _ = init_tke_calibration(datum.filename; Δz = 2.0, Δt = 1minute,
                                  calibration_kwargs(datum)...)

    push!(batch, nll)
    push!(models, nll.model)
    push!(data, nll.data)
end

batched_nll = BatchedNegativeLogLikelihood([nll for nll in batch], weights=weights)

#default_parameters = DefaultFreeParameters(batch[1].model, RiDependentTKEParameters)

default_parameters = RiDependentTKEParameters(
                                              Cᴷu⁻  = 1.2,
                                              Cᴷuᵟ  = 0.0,
                                              Cᴷc⁻  = 5.0,
                                              Cᴷcᵟ  = 0.0,
                                              Cᴷe⁻  = 0.7,
                                              Cᴷeᵟ  = 0.0,
                                              CᴷRiᶜ = -1.0,
                                              CᴷRiʷ = 0.1,
                                              Cᴰ    = 4.0,
                                              Cᴸʷ   = 1.0,
                                              Cᴸᵇ   = 1.0,
                                              Cʷu★  = 1.0,
                                              CʷwΔ  = 1.9,
                                             )

calibration = calibrate(batched_nll, default_parameters, samples=400, iterations=4);

savename = @sprintf("tke_batch_calibration_dz%d_dt%d.jld2", batched_nll.batch[1].model.grid.Δc,
                    batched_nll.batch[1].model.Δt / minute)

@save savename calibration

println("done")

filename = savename

include("mega_batch_visualization.jl")
