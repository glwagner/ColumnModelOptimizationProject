using ColumnModelOptimizationProject

@free_parameters TKEParametersToOptimize Cᴷu Cᴷc Cᴷe Cᴰ Cᴸʷ Cᴸᵇ Cʷu★

include("setup.jl")
include("utils.jl")

Δz = 2
Δt = 1
tag = "scaled-flux"
alt_tag = "scaled_flux"
samples = 500
iterations = 4

batches = OrderedDict(
               "tke-$tag-rotating-strong-stratification-mini-batch" => (
                    @sprintf("tke_calibration_%s_ekman_N²1e-7_dz%d_dt%d.jld2", alt_tag, Δz, Δt),  
                    @sprintf("tke_calibration_%s_ekman_N²1e-6_dz%d_dt%d.jld2", alt_tag, Δz, Δt)
               ),

               "tke-$tag-rotating-weak-stratification-mini-batch" => (
                    @sprintf("tke_calibration_%s_ekman_N²1e-5_dz%d_dt%d.jld2", alt_tag, Δz, Δt),  
                    @sprintf("tke_calibration_%s_ekman_N²1e-4_dz%d_dt%d.jld2", alt_tag, Δz, Δt)
               ),

               "tke-$tag-non-rotating-weak-stratification-mini-batch" => (
                    @sprintf("tke_calibration_%s_kato_N²1e-7_dz%d_dt%d.jld2", alt_tag, Δz, Δt),  
                    @sprintf("tke_calibration_%s_kato_N²1e-6_dz%d_dt%d.jld2", alt_tag, Δz, Δt)
               ),

                "tke-$tag-non-rotating-strong-stratification-mini-batch" => (
                    @sprintf("tke_calibration_%s_kato_N²1e-5_dz%d_dt%d.jld2", alt_tag, Δz, Δt),  
                    @sprintf("tke_calibration_%s_kato_N²1e-4_dz%d_dt%d.jld2", alt_tag, Δz, Δt)
                ),
)

datapath = "batching-study/tke-data"

#=
batchname = "tke-$tag-mega-batch.jld2"
memberdata = vcat([b for b in batches["tke-$tag-rotating-strong-stratification-mini-batch"]],
                  [b for b in batches["tke-$tag-rotating-weak-stratification-mini-batch"]],
                  [b for b in batches["tke-$tag-non-rotating-weak-stratification-mini-batch"]],
                  [b for b in batches["tke-$tag-non-rotating-strong-stratification-mini-batch"]])
=#

batchname = "tke-$tag-rotating-mini-batch.jld2"
memberdata = vcat([b for b in batches["tke-$tag-rotating-strong-stratification-mini-batch"]],
                  [b for b in batches["tke-$tag-rotating-weak-stratification-mini-batch"]])

#batchname = "tke-$tag-non-rotating-mini-batch.jld2"
#memberdata = vcat([b for b in batches["tke-$tag-non-rotating-strong-stratification-mini-batch"]],
#                  [b for b in batches["tke-$tag-non-rotating-weak-stratification-mini-batch"]])

#for batch in keys(batches)
#    batchname = batch * ".jld2"
#    memberdata = batches[batch]

    @show memberdata

    cases = [load(joinpath(datapath, filename), "calibration") for filename in memberdata]

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
    calibration = calibrate(batched_nll, default_parameters, samples=samples, iterations=iterations)

    println("Done.")

    @save batchname calibration
#end
