import PyPlot: plot

plot(f::AbstractField, args...; kwargs...) = plot(data(f), nodes(f), args...; kwargs...)

function removespine(side; ax=gca())
    ax.spines[side].set_visible(false)
    return nothing
end

removespines(sides...; ax=gca()) = for s in sides; removespine(s; ax=ax); end

function summarize_data(filepath; idata=[1, 2, 10, 18, 26],
                        figaxs=subplots(ncols=2, sharey=true, figsize=(8, 4)),
                        title=nothing)

    fig, axs = figaxs

    if title != nothing
        fig.suptitle(title)
    end

    font_manager = pyimport("matplotlib.font_manager")

    # Setup
    N, L = getgridparams(filepath)
    grid = UniformGrid(N, L)
    T = CellField(grid)
    U = CellField(grid)
    V = CellField(grid)
    𝒰 = CellField(grid)

    iters = iterations(filepath)
    t = times(filepath)

    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    font = font_manager.FontProperties()
    font.set_style("normal")
    font.set_weight("light")
    font.set_size("large")

    axs[2].tick_params(left=false, labelleft=false)

    for (iplot, i) = enumerate(idata)

        tlabel = @sprintf("\$ t = %.1f \$ days", t[i]/day)

        OceanTurb.set!(T, getdata("T", filepath, i))
        OceanTurb.set!(U, getdata("U", filepath, i))
        OceanTurb.set!(V, getdata("V", filepath, i))
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
