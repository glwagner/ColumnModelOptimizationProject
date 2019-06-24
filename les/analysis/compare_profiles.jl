using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz128_Nz512"

#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz128_Nz256"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz128_Nz256"
#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz128_Nz256"

filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=5, targets=(49, 249))
model = ModularKPPOptimization.ColumnModel(data, 2minute, Δ=4)
params = DefaultFreeParameters(model, WindMixingParameters)

fig, axs = visualize_realizations(data, model, params)
gcf()

#fig, axs = visualize_targets(data)
#gcf()

#=
chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-03_std1.0e-02_064.jld2"
chainpath = joinpath(chaindir, chainname)
@load chainpath chain
opt = optimal(chain)
=#
