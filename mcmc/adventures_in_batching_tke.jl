using ColumnModelOptimizationProject

@free_parameters TKEParametersToOptimize Cᴷu Cᴷc Cᴷe Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

Δz = 4
Δt = 1

memberdata = (
              @sprintf("tke_calibration_ekman_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("tke_calibration_ekman_N²1e-6_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("tke_calibration_ekman_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("tke_calibration_ekman_N²1e-4_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "tke_calibration_kato_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "tke_calibration_kato_N²1e-6_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "tke_calibration_kato_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "tke_calibration_kato_N²1e-4_dz%d_dt%d.jld2", Δz, Δt),  
            )

datapath = "batching-study/tke-data"

cases = [load(joinpath(datapath, filename), "tke_annealing") for filename in memberdata]

weights = zeros(length(cases))

println("Batch information:")
for (i, case) in enumerate(cases)
    weight = 1 / optimal(case.markov_chains[end]).error
    param = optimal(case.markov_chains[end]).param
    member = memberdata[i]

    println("")
    @show member weight param

    weights[i] = weight
end

nlls = [case.negative_log_likelihood for case in cases]
batched_nll = BatchedNegativeLogLikelihood(nlls, weights=weights)

default_parameters = DefaultFreeParameters(nlls[1].model, TKEParametersToOptimize)
annealing = calibrate(batched_nll, default_parameters, samples=1000, iterations=3);

println("Done.")

@save "tke-mega-batch.jld2" annealing
