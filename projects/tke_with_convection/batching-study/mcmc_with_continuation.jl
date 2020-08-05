using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

path(name) = joinpath("tke-data", name)

load_calibration(case_path, calibration_name="calibration") =
    load(case_path, calibration_name)

tag = "scaled-flux"
alt_tag = "scaled_flux"
mega_batch = "tke-$tag-mega-batch.jld2"

nchunks = 100
chunksize = 1000
chunks = [chunksize for i=1:nchunks]

continuation_tree = OrderedDict(
    "tke-$tag-rotating-mini-batch.jld2" => OrderedDict(

        "tke-$tag-rotating-weak-stratification-mini-batch.jld2" => (
              "tke_calibration_$(alt_tag)_ekman_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_$(alt_tag)_ekman_N²1e-6_dz2_dt1.jld2"),

        "tke-$tag-rotating-strong-stratification-mini-batch.jld2" => (
              "tke_calibration_$(alt_tag)_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_$(alt_tag)_ekman_N²1e-4_dz2_dt1.jld2")

    ),

    "tke-$tag-non-rotating-mini-batch.jld2" => OrderedDict(

        "tke-$tag-non-rotating-weak-stratification-mini-batch.jld2" => (
              "tke_calibration_$(alt_tag)_kato_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_$(alt_tag)_kato_N²1e-6_dz2_dt1.jld2"),

        "tke-$tag-non-rotating-strong-stratification-mini-batch.jld2" => (
              "tke_calibration_$(alt_tag)_kato_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_$(alt_tag)_kato_N²1e-4_dz2_dt1.jld2")

    ),
)

# First, sample from the megabatch.
mega_path = path(mega_batch)
mega_calibration = load_calibration(mega_path)

#extend_and_save!(mega_calibration, chunks, mega_path)

# Now walk breadth-first down the tree
#for (mini_batch, mini_batch_tree) in continuation_tree
#
#
mini_batch = continuation_tree.keys[1]
mini_batch_tree = continuation_tree[mini_batch]

mini_batch_continuation = continuation(mini_batch, mega_calibration, chunks)

for (mini_mini_batch, cases) in mini_batch_tree

    mini_mini_batch_continuation = continuation(mini_mini_batch, mini_batch_continuation, chunks)

    for case in cases
        case_continuation = continuation(case, mini_mini_batch_continuation, chunks)
    end

end

#end
