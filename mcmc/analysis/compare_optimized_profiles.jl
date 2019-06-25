using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
name = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz64_Nz128_e1.0e-03_dt5.0_Δ2.jld2"
filepath = joinpath(datadir, name)
@load filepath chain

data = chain.nll.data
data.targets = [481]
model = chain.nll.model
defaultparams = chain[1].param
optimalparams = optimal(chain).param

fig, axs = visualize_realizations(data, model,
                                  defaultparams, optimalparams;
                                  paramlabels=["default", "optimal"]
                                  )

gcf()
