using JLD2, PyPlot, OceanTurb, Dao,
        ColumnModelOptimizationProject, Printf

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
name = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256"
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=2, targets=(10, 30, 71))

fig, axs = visualize_targets(data)
gcf()
