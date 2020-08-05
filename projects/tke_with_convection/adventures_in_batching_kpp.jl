using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

Δz = 2
Δt = 1

memberdata = (
              @sprintf("kpp_calibration_ekman_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("kpp_calibration_ekman_N²1e-6_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("kpp_calibration_ekman_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf("kpp_calibration_ekman_N²1e-4_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "kpp_calibration_kato_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "kpp_calibration_kato_N²1e-6_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "kpp_calibration_kato_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
              @sprintf( "kpp_calibration_kato_N²1e-4_dz%d_dt%d.jld2", Δz, Δt),  
            )

datapath = "batching-study/kpp-data"

cases = [load(joinpath(datapath, filename), "kpp_calibration") for filename in memberdata]

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

default_parameters = DefaultFreeParameters(nlls[1].model, KPPWindMixingOrConvectionParameters)
kpp_calibration = calibrate(batched_nll, default_parameters, samples=1000, iterations=3);

println("Done.")

@save "kpp-mega-batch.jld2" kpp_calibration
