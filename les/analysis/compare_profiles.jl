using JLD2, PyPlot, Oceananigans, OceanTurb, ColumnModelOptimizationProject

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils

datadir = joinpath("..", "data")
filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_profiles.jld2"
filepath = joinpath(datadir, filename)

data = ColumnData(filepath)
model = ModularKPPOptimization.ColumnModel(data, minute)

defaults = ModularKPPOptimization.DefaultFreeParameters(model, ShearUnstableParameters)
