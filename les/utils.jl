using Plots, Oceananigans, Statistics, OceananigansAnalysis

include("flux_utils.jl")

removespine(side; ax=gca()) = ax.spines[side].set_visible(false)
removespines(sides...; ax=gca()) = [removespine(side, ax=ax) for side in sides]
usecmbright()

"""
    match_yaxes!(ax1, ax2)

Set the y-origin and height of `ax1` to that of `ax2`.
"""
function match_yaxes!(ax1, ax2)

    pos1 = [a for a in ax1.get_position().bounds]
    pos2 = [a for a in ax2.get_position().bounds]

    # Set y-position of pos1 to pos2
    # new position:
    #           x0      y0      width    height
    newpos = [pos1[1], pos2[2], pos1[3], pos2[4]]

    ax1.set_position(newpos)

    return nothing
end

function nice_three_plots(axs, model)
    # pcolor plots
    for ax in axs[1:3, 1]
        ax.axis("off")
        ax.set_aspect(1)
        ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    end

    # Right hand horizontal-mean plots
    for (i, ax) in enumerate(axs[1:3, 2])
        sca(ax)
        ax.tick_params(left=false, labelleft=false, right=true, labelright=true,
                       bottom=false, labelbottom=false)
        removespines("left", "top", "bottom")
        ylim(-model.grid.Lz, 0)
        ax.yaxis.set_label_position("right")
        ylabel(L"z")
    end

    for i = 1:3
        match_yaxes!(axs[i, 2], axs[i, 1])
    end

    return nothing
end

function makeplot(axs, model)

    wb = model.velocities.w * model.tracers.T
     e = turbulent_kinetic_energy(model)
     b = fluctuation(model.tracers.T)
     @. b.data *= model.constants.g * model.eos.βT

     wmax = maxabs(model.velocities.w)
     bmax = maxabs(b)

    # Top row
    sca(axs[1, 1])
    cla()
    plot_xzslice(e, cmap="YlGnBu_r")
    title(L"e")

    sca(axs[1, 2])
    cla()
    plot_hmean(e)
    title(L"\bar{e}")

    # Middle row
    sca(axs[2, 1])
    cla()
    plot_xzslice(b, cmap="RdBu_r", vmin=-bmax, vmax=bmax)
    title(L"b")

    sca(axs[2, 2])
    cla()
    plot_hmean(model.velocities.u)
    plot_hmean(model.velocities.v)

    # Bottom row
    sca(axs[3, 1])
    cla()
    plot_xzslice(model.velocities.w, cmap="RdBu_r", vmin=-wmax, vmax=wmax)
    title(L"w")

    sca(axs[3, 2])
    cla()
    plot_hmean(model.tracers.T, normalize=true, label=L"T")
    plot_hmean(wb, normalize=true, label=L"\overline{wb}")
    xlim(-1, 1)
    legend()

    nice_three_plots(axs, model)
    
    return nothing
end

function channelplot(axs, model)

    e = turbulent_kinetic_energy(model)

    umax = maxabs(model.velocities.u)
    wmax = maxabs(model.velocities.w)
    cmax = maxabs(model.tracers.S)

    # Top row
    sca(axs[1, 1])
    cla()
    plot_xzslice(e, cmap="YlGnBu_r")
    title(L"e")

    sca(axs[1, 2])
    cla()
    plot_hmean(model.velocities.v, label=L"\bar v")
    plot_hmean(√, e, label=L"\sqrt{\bar{e}}")
    #legend()

    # Middle row
    sca(axs[2, 1])
    cla()
    plot_xzslice(model.velocities.u, cmap="RdBu_r", vmin=-umax, vmax=umax)
    title(L"u")

    sca(axs[2, 2])
    cla()
    plot_hmean(model.velocities.u)

    # Bottom row
    sca(axs[3, 1])
    cla()
    plot_xzslice(log10, CellField(model.diffusivities.νₑ, model.grid), cmap="YlGnBu_r")
    title(L"log_\mathrm{10}(\nu_e)")

    sca(axs[3, 2])
    cla()
    plot_hmean(model.tracers.T, normalize=true, label=L"b")
    xlim(-1, 1)
    xlabel(L"b")

    nice_three_plots(axs, model)

    return nothing
end

