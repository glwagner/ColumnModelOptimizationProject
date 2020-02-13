using ColumnModelOptimizationProject

@free_parameters TKEParametersToOptimize Cᴷu Cᴷc Cᴷe Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

Δz = 2
Δt = 1

batches = OrderedDict(
               "tke-rotating-strong-stratification-mini-batch" => (
                    @sprintf("tke_calibration_simple_ekman_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
                    @sprintf("tke_calibration_simple_ekman_N²1e-6_dz%d_dt%d.jld2", Δz, Δt)
               ),

               "tke-rotating-weak-stratification-mini-batch" => (
                    @sprintf("tke_calibration_simple_ekman_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
                    @sprintf("tke_calibration_simple_ekman_N²1e-4_dz%d_dt%d.jld2", Δz, Δt)
               ),

               "tke-non-rotating-weak-stratification-mini-batch" => (
                    @sprintf("tke_calibration_simple_kato_N²1e-7_dz%d_dt%d.jld2", Δz, Δt),  
                    @sprintf("tke_calibration_simple_kato_N²1e-6_dz%d_dt%d.jld2", Δz, Δt)
               ),

                "tke-non-rotating-strong-stratification-mini-batch" => (
                    @sprintf("tke_calibration_simple_kato_N²1e-5_dz%d_dt%d.jld2", Δz, Δt),  
                    @sprintf("tke_calibration_simple_kato_N²1e-4_dz%d_dt%d.jld2", Δz, Δt)
                ),
)

datapath = "batching-study/tke-data"

batchname = "tke-mega-batch.jld2"
memberdata = vcat([b for b in batches["rotating-strong-stratification"]],
                  [b for b in batches["rotating-weak-stratification"]],
                  [b for b in batches["non-rotating-weak-stratification"]],
                  [b for b in batches["non-rotating-strong-stratification"]])

#batchname = "tke-rotating-mini-batch.jld2"
#memberdata = vcat([b for b in batches["rotating-strong-stratification"]],
#                  [b for b in batches["rotating-weak-stratification"]])

#batchname = "tke-non-rotating-mini-batch.jld2"
#memberdata = vcat([b for b in batches["non-rotating-strong-stratification"]],
#                  [b for b in batches["non-rotating-weak-stratification"]])

#for batch in keys(batches)
#batch = "rotating-strong-stratification"
#    batchname = batch * ".jld2"
#    memberdata = batches[batch]

    @show memberdata

    cases = [load(joinpath(datapath, filename), "tke_calibration") for filename in memberdata]

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
    calibration = calibrate(batched_nll, default_parameters, samples=1000, iterations=3);

    println("Done.")

    @save batchname calibration
#end
