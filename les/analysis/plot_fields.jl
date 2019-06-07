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

    sca(axs[1, 1])
    plot_xzslice(u, cmap="RdBu_r", vmin=-umax, vmax=umax, shading="gouraud")
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
    plot_hmean(wθ, normalize=true, label=L"\overline{wT}")
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

    axs[1, 2].tick_params(right=true, labelright=true, left=false, labelleft=false, bottom=false, labelbottom=false,
                            top=true, labeltop=true)

    axs[2, 2].tick_params(right=true, labelright=true, left=false, labelleft=false,
                            bottom=false, labelbottom=false)


    removespines("left", "top", ax=axs[2, 2])

    axs[1, 2].xaxis.set_label_position("top")

    return nothing
end

datadir = "data"
#filename = "simple_flux_Fb1e-08_Fu1e-04_Nsq1e-06_Lz64_Nz256_fields.jld2"
#filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_fields.jld2"
#filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_fields.jld2"
filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_fields.jld2"
filepath = joinpath(datadir, filename)

g = Grid(filepath)
iters = get_iters(filepath)

gridspec = Dict("width_ratios"=>[Int(g.Lx/g.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=2, figsize=(8, 6), gridspec_kw=gridspec)

for (ii, i) in enumerate(iters)

    for ax in axs
        sca(ax)
        cla()
    end

    makeplot(filepath, axs, g, i, wclip=0.5, uclip=0.5)
    gcf()

    savefig(@sprintf("%s_%04d.png", filename, ii), dpi=480)
end

gcf()
