using JLD2, PyPlot, Oceananigans, OceananigansAnalysis, Statistics

parameters = Dict(:free_convection => Dict(:Fb=>3.39e-8, :Fu=>0.0,     :f=>1e-4, :N²=>1.96e-5),
                  :wind_stress     => Dict(:Fb=>0.0,     :Fu=>9.66e-5, :f=>0.0,  :N²=>9.81e-5))

include("file_wrangling.jl")

datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/les/data"
#filename = "free_convection_Nx256_Nz256.jld2"
filename = "free_convection_Nx128_Nz128.jld2"
filepath = joinpath(datadir, filename)

grid = Grid(filepath)
iters = get_iters(filepath)
h = zeros(length(iters))
t = zeros(length(iters))
iter = iters[end]

close("all")
fig, axs = subplots(ncols=2, figsize=(14, 6))

u = FaceFieldX(get_snapshot(filepath, :u, iter), grid)
v = FaceFieldY(get_snapshot(filepath, :v, iter), grid)
w = FaceFieldZ(get_snapshot(filepath, :w, iter), grid)
θ = CellField(get_snapshot(filepath, :T, iter), grid)
θ′ = fluctuation(θ)

function mixed_layer_depth(θ)
    Nz = θ.grid.Nz
    T = dropdims(mean(view(θ.data, 1:Nz, 1:Nz, :), dims=(1, 2)), dims=(1, 2))

    T[1] = T[2]
    δT = T[end-2] - T[end-1]
    T[end] = T[end-1] - δT
    Tz = (T[1:end-1] - T[2:end]) / θ.grid.Δz

    i_entrainment = argmax(Tz)

    return -θ.grid.zF[i_entrainment]
end

sca(axs[1])
plot_xzslice(w, slice=10, cmap="RdBu_r", shading="flat")#, vmin=-0.06, vmax=0.06)

sca(axs[2])
plot_xyslice(w, slice=20, cmap="RdBu_r", shading="flat")#, vmin=-0.06, vmax=0.06)

fig2, axs2 = subplots(ncols=2)

for (i, iter) in enumerate(iters)
    θ = CellField(get_snapshot(filepath, :T, iter), grid)
    if i > 1
        h[i] = mixed_layer_depth(θ)
        t[i] = get_time(filepath, iter)
    end

    T = dropdims(havg(θ), dims=(1, 2))
    T[1] = T[2]
    δT = T[end-2] - T[end-1]
    T[end] = T[end-1] - δT
    Tz = (T[1:end-1] - T[2:end]) / θ.grid.Δz
    
    sca(axs2[1])
    plot(T[2:end-1], θ.grid.zC)

    sca(axs2[2])
    plot(Tz, θ.grid.zF)
end

fig3, axs3 = subplots()
axs3.invert_yaxis()

Fb = parameters[:free_convection][:Fb]
N² = parameters[:free_convection][:N²]
h_analytic = @. sqrt(2.8 * Fb * t / N²)

plot(t/day, h, "o-")
plot(t/day, h_analytic, "--")

xlabel("\$ t \$ (days)")
ylabel("\$ h \$ (m)")
