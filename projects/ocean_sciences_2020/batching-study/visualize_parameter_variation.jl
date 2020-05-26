using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

tag = "scaled-flux"
alt_tag = "scaled_flux"
mega_batch = "tke-$tag-mega-batch.jld2"

continuation_tree = OrderedDict(
    "tke-$tag-rotating-mini-batch-continuation.jld2" => OrderedDict(

        "tke-$tag-rotating-weak-stratification-mini-batch-continuation.jld2" => (
              "tke_calibration_$(alt_tag)_ekman_N²1e-7_dz2_dt1-continuation.jld2",
              "tke_calibration_$(alt_tag)_ekman_N²1e-6_dz2_dt1-continuation.jld2"),

        "tke-$tag-rotating-strong-stratification-mini-batch-continuation.jld2" => (
              "tke_calibration_$(alt_tag)_ekman_N²1e-5_dz2_dt1-continuation.jld2",
              "tke_calibration_$(alt_tag)_ekman_N²1e-4_dz2_dt1-continuation.jld2")

    ),

    "tke-$tag-non-rotating-mini-batch-continuation.jld2" => OrderedDict(

        "tke-$tag-non-rotating-weak-stratification-mini-batch-continuation.jld2" => (
              "tke_calibration_$(alt_tag)_kato_N²1e-7_dz2_dt1-continuation.jld2",
              "tke_calibration_$(alt_tag)_kato_N²1e-6_dz2_dt1-continuation.jld2"),

        "tke-$tag-non-rotating-strong-stratification-mini-batch-continuation.jld2" => (
              "tke_calibration_$(alt_tag)_kato_N²1e-5_dz2_dt1-continuation.jld2",
              "tke_calibration_$(alt_tag)_kato_N²1e-4_dz2_dt1-continuation.jld2")

    ),
)

function get_optimal(name, case)
    casepath = joinpath("tke-data", case)
    calibration = load(casepath, "calibration")
    chain = calibration.markov_chains[end]

    initial_error = calibration.markov_chains[1][1].error
    optimal_error = optimal(chain).error

    return getproperty(optimal(chain).param, name), optimal_error / initial_error
end

function plot_paths(ax, parameter_name, mega_batch, continuation_tree)

    levels = 4
    cases = 8
    paths = zeros(levels, cases)
    errors = zeros(levels, cases)
    mega_optimal, mega_error = get_optimal(parameter_name, mega_batch)

    paths[1, :] .= mega_optimal
    errors[1, :] .= mega_error

    for (i, mini_batch) in enumerate(continuation_tree.keys)
        level = 1
        breadth = 4 #Int(cases/2^level)

        mini_optimal, mini_error = get_optimal(parameter_name, mini_batch)

        i1 = (i-1) * breadth + 1
        i2 = i * breadth
        paths[level+1, i1:i2] .= mini_optimal
        errors[level+1, i1:i2] .= mini_error

        mini_batch_tree = continuation_tree[mini_batch]

        for (j, mini_mini_batch) in enumerate(mini_batch_tree.keys)
            level = 2
            inner_breadth = 2 #Int(cases/2^level)

            mini_mini_optimal, mini_mini_error = get_optimal(parameter_name, mini_mini_batch)

            j1 = (i-1) * breadth + (j-1) * inner_breadth + 1 
            j2 = (i-1) * breadth + j*inner_breadth

            paths[level+1, j1:j2] .= mini_mini_optimal
            errors[level+1, j1:j2] .= mini_mini_error

            cases = mini_batch_tree[mini_mini_batch]

            for (k, case) in enumerate(cases)
                level = 3
                case_optimal, case_error = get_optimal(parameter_name, case)

                kcase = (i-1) * breadth + (j-1) * inner_breadth + k
                paths[level+1, kcase] = case_optimal
                errors[level+1, kcase] = case_error
            end
        end
    end

    sca(ax)

    for i = 1:8
        plot(1:4, paths[:, i], "k-", linewidth=0.5, alpha=0.6, zorder=1)
    end

    return paths, errors
end

function reduce_tree(tree)
    reduced = [[0.0], zeros(2), zeros(4), zeros(8)]

    reduced[1] .= tree[4, 1]
    reduced[2] .= tree[4, [1, 5]]
    reduced[3] .= tree[4, [1, 3, 5, 7]]
    reduced[4] .= tree[4, :]

    return reduced
end

#parameter_names = [:Cᴷu, :Cᴷc, :Cᴷe, :Cᴰ, :Cᴸᵇ, :Cᴸʷ, :Cʷu★]
parameter_names = [:Cᴷc, :Cᴸᵇ]


caselabels = [
              "r, v weak",
              "r, weak",
              "r, strong",
              "r, v strong",
              "n, v weak",
              "n, weak",
              "n, med",
              "n, strong",
              #=
              "r, \$10^{-7}\$",
              "r, \$10^{-6}\$",
              "r, \$10^{-5}\$",
              "r, \$10^{-4}\$",
              "n, \$10^{-7}\$",
              "n, \$10^{-6}\$",
              "n, \$10^{-5}\$",
              "n, \$10^{-4}\$",
              =#
             ]

miniminimarkers = ["d", "^", "s", "h"]

miniminilabels = [
                  "rotating, weak \$N^2\$", 
                  "rotating, strong \$N^2\$", 
                  "non-rotating, weak \$N^2\$", 
                  "non-rotating, strong \$N^2\$", 
                 ]

minimarkers = ["v", "p"]

minilabels = [
              "rotating",
              "non-rotating",
             ]

ms = 8
leg_fs = 10
leg_ms = 0.8
cmap = get_cmap("YlGnBu")

function error2color(error) 
    norm = 1
    shift = 0.4
    shifted = shift + error / (1 - shift)

    return cmap(shifted * norm)
end

close("all")
fig, axs = subplots(ncols=length(parameter_names), figsize=(9.5, 3.5))

for (i, ax) in enumerate(axs)
    sca(ax)
    removespines("top", "right", "bottom")
    ax.tick_params(bottom=false)
    ylabel(parameter_latex_guide[parameter_names[i]])
end

for (k, name) in enumerate(parameter_names)
    paths, errors = plot_paths(axs[k], name, mega_batch, continuation_tree)

    reduced_paths = reduce_tree(paths)
    errors = reduce_tree(errors)

    sca(axs[k])

    plot(1, paths[1, 1], "*", color="k", markersize=16, alpha=0.8, zorder=2, 
         label="mega-batch")

    # Mini-batch optima
    for (j, i) = enumerate([1, 5])
        plot(2, paths[2, i], linestyle="None", marker=minimarkers[j],
             color = error2color(errors[2][j]),
             markersize=10, alpha=0.6, label=minilabels[j], zorder=2)
    end

    # Mini-mini-batch optima
    for (j, i) = enumerate([1, 3, 5, 7])
        plot(3, paths[3, i], linestyle="None", marker=miniminimarkers[j], 
             color = error2color(errors[3][j]),
             markersize=8, alpha=0.6, label=miniminilabels[j], zorder=2)
    end

    # Individual optima
    for i = 1:8
        plot(4, paths[4, i], "o",
             markersize=6, alpha=0.8,
             color = error2color(errors[4][i]),
             zorder=2)

        if k == 1 && i == 3 # r, strong
            ytext = paths[4, i] + 0.08
        elseif k == 1 && i == 5 # n, v weak
            ytext = paths[4, i] - 0.13
        elseif k == 1 && i == 6 # n, weak
            ytext = paths[4, i] - 0.06
        elseif k == 1 && i == 7 # n, med
            ytext = paths[4, i] + 0.00
        elseif k == 1 && i == 8 # n, med
            ytext = paths[4, i] + 0.04

        elseif k == 2 && i == 2 # r, weak
            ytext = paths[4, i] + 0.05

        else
            ytext = paths[4, i]
        end

        text(4.1, ytext, caselabels[i], size=8, verticalalignment="center", 
             color = error2color(errors[4][i]), #color=defaultcolors[i], 
             horizontalalignment="left", alpha=0.8)
    end

    k == 1 && legend(fontsize=leg_fs, loc=2, markerscale=leg_ms, bbox_to_anchor=(0.0, 0.03, 1, 1))
        
    #legend(fontsize=leg_fs, loc=3, markerscale=leg_ms, bbox_to_anchor=(-0.03, 0, 1, 1))
end

sca(axs[1])
xticks((1, 2, 3, 4), 
       ("8-case \n calibration", "4-case \n calibration", "2-case \n calibration", "single case \n calibration"), fontsize=8)
ylim(-0.1, 2.0)

sca(axs[2])
xticks((1, 2, 3, 4), 
       ("8-case \n calibration", "4-case \n calibration", "2-case \n calibration", "single case \n calibration"), fontsize=8)

# hobo colormap
x1 = 1.2
x2 = 2.37
r = x2 - x1

e = 0.0:0.1:1.0
n = length(e)
d = r / (n-1)

for (i, ei) in enumerate(e)
    plot(x1 + r * ei, 1.0, "s", color=error2color(ei), markersize=8)
end

for ei in [0.0, 0.5, 1.0]
    text(x1 + r * ei, 0.75, @sprintf("%.1f", ei), horizontalalignment="center")
end

text(x1+r*0.5, 1.15, L"\mathcal{L}(C_\star) / \mathcal{L}(C_i)", horizontalalignment="center")

tight_layout()

savefig("parameter-continuation-viz.png", dpi=480)
