using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128"

filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=1, targets=[9, 145])
model = ModularKPPOptimization.ColumnModel(data, 5minute, Δ=8)
defaultparams = DefaultFreeParameters(model, WindMixingParameters)

legendkw = Dict(
    :markerscale=>1.2, :fontsize=>10,
    :loc=>"bottom right", :bbox_to_anchor=>(0.4, 0.5),
    :frameon=>true, :framealpha=>1.0)

#fig, axs = visualize_targets(data; legendkwargs=legendkw)
fig, axs = visualize_realization(defaultparams, model, data; legendkwargs=legendkw)
tight_layout()
gcf()

savefig("/Users/gregorywagner/Desktop/data_comparison_example.png", dpi=480)
