using JLD2, PyPlot, OceanTurb, Dao,
        ColumnModelOptimizationProject, Printf

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
#name = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256"
#name = "simple_flux_Fb1e-08_Fu-1e-04_Nsq2e-06_Lz64_Nz256"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
#name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz128_Nz512"
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=3, targets=(13, 97))

#fig, axs = visualize_targets(data)
#gcf()

chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-03_std1.0e-02_064.jld2"
chainpath = joinpath(chaindir, chainname)
@load chainpath chain
opt = optimal(chain)

fig, axs = visualize_realizations(data, chain.nll.model, opt.param, chain[1].param)
gcf()
