using JLD2, PyPlot, Oceananigans, OceanTurb

@use_pyplot_utils

include("file_wrangling.jl")

datadir = "data"
filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_profiles.jld2"
filepath = joinpath(datadir, filename)

grid = Grid(filepath)
iters = get_iters(filepath)

clf()
fig, axs = subplots(ncols=2)

for ii in (5, 9, 13) #(2, 5, 7, 9, 11, 13)
    iter = iters[ii]
    U = get_oceanturb_snapshot(filepath, :U, iter)
    V = get_oceanturb_snapshot(filepath, :V, iter)
    T = get_oceanturb_snapshot(filepath, :T, iter)

    sca(axs[1])
    plot(U, "-")
    plot(V, "--")

    sca(axs[2])
    plot(T)
end

sca(axs[1])
removespines("top", "right")

sca(axs[2])
removespines("top", "right", "left")
axs[2].tick_params(left=false, labelleft=false)

gcf()
