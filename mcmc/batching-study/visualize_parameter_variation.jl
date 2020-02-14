using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

tag = "surface-value"
alt_tag = "surface_tke_value"
mega_batch = "tke-$tag-mega-batch.jld2"

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

function get_optimal(name, case)
    casepath = joinpath("tke-data", case)
    calibration = load(casepath, "calibration")
    chain = calibration.markov_chains[end]
    return getproperty(optimal(chain).param, name)
end

fig, axs = subplots()
removespines("top", "right", "bottom")
#xticks(1:4, jjj

levels = 4
cases = 8
paths = zeros(levels, cases)

parameter_name = :Cᴷc
mega_optimal = get_optimal(parameter_name, calibrated[1])

paths[1, :] .= mega_optimal

for (i, mini_batch) in enumerate(continuation_tree.keys)
    level = 1
    breadth = 4 #Int(cases/2^level)
    mini_optimal = get_optimal(parameter_name, mini_batch)

    i1 = (i-1) * breadth + 1
    i2 = i * breadth
    paths[2, i1:i2] .= mini_optimal

    mini_batch_tree = continuation_tree[mini_batch]

    for (j, mini_mini_batch) in enumerate(mini_batch_tree.keys)
        level = 2
        inner_breadth = 2 #Int(cases/2^level)

        mini_mini_optimal = get_optimal(parameter_name, mini_mini_batch)

        j1 = (i-1) * breadth + (j-1) * inner_breadth + 1 
        j2 = (i-1) * breadth + j*inner_breadth

        paths[3, j1:j2] .= mini_mini_optimal

        cases = mini_batch_tree[mini_mini_batch]

        for (k, case) in enumerate(cases)
            level = 3

            case_optimal = get_optimal(parameter_name, case)

            @show case
            @show kcase = (i-1) * breadth + (j-1) * inner_breadth + k

            paths[4, kcase] = case_optimal
        end
    end
end

paths ./= mega_optimal

for i = 1:8
    plot(1:4, paths[:, i], "k-", linewidth=0.5, alpha=0.6)

    #plot(1, paths[3, i], "*", color="k", markersize=10, alpha=0.6)
    #plot(2, paths[3, i], "*", color="k", markersize=10, alpha=0.6)
    #plot(3, paths[3, i], "*", color="k", markersize=10, alpha=0.6)
    plot(4, paths[4, i], "*", color=defaultcolors[i], markersize=10, alpha=0.6)
end

#=
plot(1, mega_optimal, "*")

for case in calibrated[2:3]

    paths
    plot(2, mini_optimal, "*")
end

for case in calibrated[4:7]
    mini_mini_optimal = get_optimal(parameter_name, case)
    plot(3, mini_mini_optimal, "*")
end

for case in calibrated[8:15]
    case_optimal = get_optimal(parameter_name, case)
    plot(4, case_optimal, "*")
end
=#
