function summarize_data(filepath; idata=[1, 2, 10, 18, 26],
                        figaxs=subplots(ncols=2, sharey=true, figsize=(8, 4)),
                        title=nothing)

    fig, axs = figaxs

    if title != nothing
        fig.suptitle(title)
    end

    font_manager = pyimport("matplotlib.font_manager")

    # Setup
    N, L = get_grid_params(filepath)
    grid = UniformGrid(N, L)
    U = CellField(grid)
    V = CellField(grid)
    𝒰 = CellField(grid) # speed
    T = CellField(grid)
    S = CellField(grid)

    iters = get_iterations(filepath)
    t = get_times(filepath)

    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    font = font_manager.FontProperties()
    font.set_style("normal")
    font.set_weight("light")
    font.set_size("large")

    axs[2].tick_params(left=false, labelleft=false)

    iters = iterations(filepath)

    for (iplot, i) = enumerate(idata)

        iter = iters[i]

        tlabel = @sprintf("\$ t = %.1f \$ days", t[i]/day)

        OceanTurb.set!(U, get_data("U", filepath, iter))
        OceanTurb.set!(V, get_data("V", filepath, iter))
        OceanTurb.set!(T, get_data("T", filepath, iter))
        OceanTurb.set!(S, get_data("S", filepath, iter))

        OceanTurb.set!(𝒰, sqrt.(U.data.^2 + V.data.^2))

        sca(axs[1])
        plot(T, label=tlabel, color=defaultcolors[iplot])
        removespines("top", "right")
        xlabel(L"T")
        ylabel("\$ z \$ (m)")
        legend(fontsize=10)

        sca(axs[2])
        plot(U, "--", alpha=0.5, linewidth=1, color=defaultcolors[iplot])
        plot(V, ":", alpha=0.5, linewidth=1, color=defaultcolors[iplot])
        plot(𝒰, "-", label=tlabel, color=defaultcolors[iplot])
        removespines("top", "left", "right")
        xlabel(L"U, V, \, \mathrm{and} \, \sqrt{U^2 + V^2}")

        legend(fontsize=10, loc=4)
    end

    return fig, axs
end

function maxvariance(data, fldname)

    maximum_variance = 0.0

    for target in data.targets
        fld = getproperty(data, fldname)[target]
        fldmean = mean(fld.data)
        variance = 0
        for j in eachindex(fld)
            variance += (fld[j] - fldmean)^2 * Δf(fld, j)
        end
        maximum_variance = max(maximum_variance, variance)
    end

    return maximum_variance
end
