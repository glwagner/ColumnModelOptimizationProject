using JLD2, PyPlot, Oceananigans, OceananigansAnalysis, Printf

include("file_wrangling.jl")
include("../plot_utils.jl")

function make_xz_plot(filepath, axs, grid, iter)
    u = FaceFieldX(get_snapshot(filepath, :u, iter), grid)
    v = FaceFieldY(get_snapshot(filepath, :v, iter), grid)
    w = FaceFieldZ(get_snapshot(filepath, :w, iter), grid)
    θ = CellField(get_snapshot(filepath, :θ, iter), grid)

    θ′ = fluctuation(θ)
    e = turbulent_kinetic_energy(u, v, w)
    @show emax = maximum(log10, e.data)
    @show θmax = maximum(abs, θ′.data)
    @show umax = maximum(abs, u.data)

    ymin = -75

    sca(axs[1])
    ylim(ymin, 0)

    plot_xzslice(u, slice=10, cmap="RdBu_r", shading="gouraud", vmin=-0.06, vmax=0.06)
    title("horizontal velocity")
    xlabel(L"x \, (\mathrm{m})")
    ylabel(L"z \, (\mathrm{m})")

    sca(axs[2])
    ylim(ymin, 0)
    xlim(19.4, 19.7)
    plot_hmean(θ, label=L"V")
    xlabel("Temperature  (\$ {}^\\circ \\mathrm{C} \$)")
    ylabel(L"z \, (\mathrm{m})")

    removespines("top", "bottom", "right", "left", ax=axs[1])


    match_yaxes!(axs[2], axs[1])
    axs[2].yaxis.set_label_position("right")
    axs[2].tick_params(right=true, labelright=true, left=false, labelleft=false)
    removespines("left", "top", ax=axs[2])

    return nothing
end

function make_xy_plot(filepath, axs, grid, iter)
    u = FaceFieldX(get_snapshot(filepath, :u, iter), grid)
    v = FaceFieldY(get_snapshot(filepath, :v, iter), grid)
    w = FaceFieldZ(get_snapshot(filepath, :w, iter), grid)

    e = turbulent_kinetic_energy(u, v, w)
    @show emax = maximum(e.data)

    sca(axs[1])
    slice = 41
    @show grid.zC[slice]
    plot_xyslice(e, slice=slice, cmap="YlGnBu_r", shading="gouraud", vmax=0.001)
    title("turbulent kinetic energy at \$z = - 10 \\, \\mathrm{m} \$")

    sca(axs[2])
    slice = 129
    @show grid.zC[slice]
    plot_xyslice(e, slice=slice, cmap="YlGnBu_r", shading="gouraud", vmax=0.0005)
    title("turbulent kinetic energy at \$z = -32 \\, \\mathrm{m} \$")

    for ax in axs
        ax.set_aspect(1)
        sca(ax)
        xlabel(L"x \, (\mathrm{m})")
        ylabel(L"y \, (\mathrm{m})")
        removespines("top", "bottom", "right", "left")
    end

    axs[2].yaxis.set_label_position("right")
    axs[2].tick_params(left=false, labelleft=false, right=true, labelright=true)

    return nothing
end

datadir = "data"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz128_Nz512"
filepath = joinpath(@__DIR__, "..", datadir, name * "_fields.jld2")

g = Grid(filepath)
@show iters = get_iters(filepath)

#=
gridspec = Dict("width_ratios"=>[Int(g.Lx/g.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=1, figsize=(8, 3), gridspec_kw=gridspec)
make_xz_plot(filepath, axs, g, iters[end])
tight_layout()
gcf()

savefig("xz_wind_mixing.png", dpi=480)
=#

fig, axs2 = subplots(ncols=2, figsize=(10, 3))
make_xy_plot(filepath, axs2, g, iters[end])
gcf()
tight_layout()
savefig("xy_wind_mixing.png", dpi=480)
