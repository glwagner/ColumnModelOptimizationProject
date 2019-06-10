using JLD2, PyPlot, Oceananigans, OceananigansAnalysis, Printf

include("file_wrangling.jl")
include("../plot_utils.jl")

resize_plane(u, grid) = repeat(reshape(u, grid.Nx+2, 1, grid.Nz), 1, 3, 1)

get_plane_snapshot(Field, fld, i, filepath, grid) =
    Field(resize_plane(get_snapshot(filepath, fld, i), grid), grid)

function makeplot(filepath, axs, grid, i; wclip=0.5, uclip=0.5)

    u = get_plane_snapshot(FaceFieldX, :uxz, i, filepath, grid)
    v = get_plane_snapshot(FaceFieldY, :vxz, i, filepath, grid)
    w = get_plane_snapshot(FaceFieldZ, :wxz, i, filepath, grid)
    θ = get_plane_snapshot(CellField,  :θxz, i, filepath, grid)

    umax = maximum(abs, u.data.parent)
    wmax = maximum(abs, w.data.parent)

    sca(axs[1])
    plot_xzslice(u, cmap="RdBu_r") #, vmin=-umax, vmax=umax, shading="gouraud")
    ylabel(L"z \, (\mathrm{m})")

    sca(axs[2])
    plot_xzslice(w, cmap="RdBu_r") #, vmin=-wmax, vmax=wmax, shading="gouraud")
    ylabel(L"z \, (\mathrm{m})")
    xlabel(L"x \, (\mathrm{m})")

    for ax in axs
        removespines("top", "bottom", "right", "left", ax=ax)
        ax.set_aspect(1)
    end

    axs[1].tick_params(bottom=false, labelbottom=false)

    return nothing
end

datadir = "data"
name = "simple_flux_Fb5e-09_Fu-1e-04_Nsq5e-06_Lz128_Nz256"
filepath = joinpath(@__DIR__, "..", datadir, name * "_planes.jld2")

g = XZGrid(filepath)
iters = get_iters(filepath)

fig, axs = subplots(nrows=2, figsize=(8, 6))

u = get_snapshot(filepath, :uxz, 0)

for (ii, i) in enumerate(iters)

    for ax in axs
        sca(ax)
        cla()
    end

    @show iters[ii]
    makeplot(filepath, axs, g, iters[ii], wclip=0.5, uclip=0.5)

    savefig(@sprintf("%s_%04d.png", name, ii), dpi=480)
end
