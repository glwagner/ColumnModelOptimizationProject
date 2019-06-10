using JLD2, PyPlot, Oceananigans, OceananigansAnalysis, Printf

include("file_wrangling.jl")
include("../plot_utils.jl")

#Oceananigans.data(f::Field{A}) where A<:Array = f.data

function makeplot(filepath, axs, g, i; wclip=0.25, uclip=0.25)
    u = FaceFieldX(get_snapshot(filepath, :u, i), g)
    v = FaceFieldY(get_snapshot(filepath, :v, i), g)
    w = FaceFieldZ(get_snapshot(filepath, :w, i), g)
    θ = CellField(get_snapshot(filepath, :θ, i), g)

    e = turbulent_kinetic_energy(u, v, w)
    wθ = w * θ

    wmax = maximum(abs, w.data) * wclip
    umax = maximum(abs, u.data) * uclip

    @show θmax = maximum(θ.data)
    @show θmin = minimum(θ.data)

    sca(axs[1, 1])
    #plot_xzslice(u, cmap="RdBu_r", vmin=-umax, vmax=umax, shading="gouraud")
    plot_xzslice(θ, cmap="YlGnBu_r", vmin=19.98, vmax=19.985)
    ylabel(L"z \, (\mathrm{m})")
    title(L"u")

    sca(axs[2, 1])
    plot_xzslice(w, cmap="RdBu_r", vmin=-wmax, vmax=wmax, shading="gouraud")
    ylabel(L"z \, (\mathrm{m})")
    xlabel(L"x \, (\mathrm{m})")
    title(L"w")

    sca(axs[1, 2])
    plot_hmean(u, label=L"U")
    plot_hmean(v, label=L"V")
    plot_hmean(√, e, label=L"\sqrt{\bar e}")
    legend()

    sca(axs[2, 2])
    plot_hmean(θ, normalize=true, label=L"\bar T")
    #plot_hmean(wθ, normalize=true, label=L"\overline{wT}")
    legend()

    for ax in axs[1:2, 1]
        removespines("top", "bottom", "right", "left", ax=ax)
        ax.set_aspect(1)
    end

    for ax in axs[1:2, 2]
        ax.yaxis.set_label_position("right")
        ylabel(L"z \, (m)")
    end

    match_yaxes!(axs[1, 2], axs[1, 1])
    match_yaxes!(axs[2, 2], axs[2, 1])

    axs[1, 1].tick_params(bottom=false, labelbottom=false)

    axs[1, 2].tick_params(right=true, labelright=true, left=false, labelleft=false,
                            bottom=false, labelbottom=false,
                            top=true, labeltop=true)

    axs[2, 2].tick_params(right=true, labelright=true, left=false, labelleft=false)

    removespines("left", "bottom", ax=axs[1, 2])
    removespines("left", "top", ax=axs[2, 2])

    axs[1, 2].xaxis.set_label_position("top")

    return nothing
end

datadir = "data"
#name = "simple_flux_Fb5e-09_Fu-1e-04_Nsq1e-06_Lz128_Nz256"
name = "simple_flux_Fb1e-09_Fu0e+00_Nsq1e-06_Lz128_Nz256"
filepath = joinpath(@__DIR__, "..", datadir, name * "_fields.jld2")

g = Grid(filepath)
@show iters = get_iters(filepath)

gridspec = Dict("width_ratios"=>[Int(g.Lx/g.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=2, figsize=(8, 6), gridspec_kw=gridspec)

for (ii, i) in enumerate(iters)

    for ax in axs
        sca(ax)
        cla()
    end

    makeplot(filepath, axs, g, i, wclip=0.5, uclip=0.5)
    gcf()

    savefig(@sprintf("%s_%04d.png", name, ii), dpi=480)
end

i = 30
gridspec = Dict("width_ratios"=>[Int(g.Lx/g.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=2, figsize=(8, 6), gridspec_kw=gridspec)
makeplot(filepath, axs, g, iters[i], wclip=0.5, uclip=0.5)
gcf()
