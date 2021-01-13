using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
#LESbrary_path = "/home/glwagner/BoundaryLayerTurbulenceSimulations/idealized/data"

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
            eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities(Cᴷu⁺=0.1),
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
    nll, _ = init_tke_calibration(datum.filename; Δz = 2.0, Δt = minute / 2,
                                  calibration_kwargs(datum)...)

    push!(batch, nll)
    push!(models, nll.model)
    push!(data, nll.data)
end

batched_nll = BatchedNegativeLogLikelihood([nll for nll in batch], weights=weights)

#default_parameters = DefaultFreeParameters(batch[1].model, RiDependentTKEParameters)
#
#=
default_parameters = RiDependentTKEParameters(
                                              CᴷRiʷ = 0.11,
                                              CᴷRiᶜ = -0.3,
                                              Cᴷu⁻  = 4.1,
                                              Cᴷu⁺  = 0.06,
                                              Cᴷc⁻  = 0.8,
                                              Cᴷc⁺  = 0.06,
                                              Cᴷe⁻  = 3.7,
                                              Cᴷe⁺  = 0.001,
                                              Cᴰ    = 0.8,
                                              Cᴸᵇ   = 2.1,
                                              Cʷu★  = 4.0,
                                              CʷwΔ  = 4.9,
                                             )
=#

default_parameters = RiDependentTKEParameters(
                                              CᴷRiʷ = 0.5313, # 0.1617,  # 0.1274,  # 0.26,
                                              CᴷRiᶜ = -0.7300, # -0.2945, # -0.8256, # -0.83,
                                              Cᴷu⁻  = 0.7062, # 4.4864,  # 0.4191,  # 3.12,
                                              Cᴷu⁺  = 0.0071, # 0.0037,  # 0.0405,  # 0.0104,
                                              Cᴷc⁻  = 4.5015, # 3.4370,  # 1.9687,  # 3.7515,
                                              Cᴷc⁺  = 0.0283, # 0.0371,  # 0.01,    # 0.0966,
                                              Cᴷe⁻  = 1.2482, # 2.1622,  # 1.3169,  # 1.6082,
                                              Cᴷe⁺  = 0.0323, # 0.0801,  # 0.072,   # 0.0996,
                                              Cᴰ    = 0.7884, # 0.6336,  # 0.4441,  # 2.6418,
                                              Cᴸᵇ   = 5.8012, # 2.7066,  # 5.9172,  # 5.7232,
                                              Cʷu★  = 4.1025, # 3.0373,  # 4.4074,  # 3.1845,
                                              CʷwΔ  = 0.8092, # 4.1925,  # 5.4895,  # 0.4512,
                                             )

for i = 1:10
    global default_parameters

    calibration = calibrate(batched_nll, default_parameters, samples=1000, iterations=4);
    
    savename = @sprintf("tke_batch_calibration_convection_refine_dz%d_dt%d.jld2", batched_nll.batch[1].model.grid.Δc,
                        batched_nll.batch[1].model.Δt / minute)
    
    @save savename calibration

    println("done")
    
    filename = savename
    
    include("mega_batch_visualization.jl")

    opt = optimal(calibration.markov_chains[end])
    default_parameters = opt.param
end
