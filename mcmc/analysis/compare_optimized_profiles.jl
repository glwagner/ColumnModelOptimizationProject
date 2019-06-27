using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
#name = "mcmc_shape_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-06_Lz64_Nz128_e1.0e-05_dt5.0_Δ2.jld2"

name = "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e2.0e-03_dt5.0_Δ2.jld2"
#name = "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_e5.0e-03_dt5.0_Δ2.jld2"
#name = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e2.0e-03_dt5.0_Δ2.jld2"
#name = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_e2.0e-03_dt5.0_Δ2.jld2"
#name = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-06_Lz64_Nz128_e2.0e-03_dt5.0_Δ2.jld2"

filepath = joinpath(datadir, name)
@load filepath chain

data = chain.nll.data
model = chain.nll.model
defaultparams = chain[1].param
optimalparams = optimal(chain).param

fig, axs = visualize_realizations(data, model,
                                  optimalparams, defaultparams;
                                  paramlabels=["optimal", "default"]
                                  )

gcf()


#
# Other data
#

#=
lesdatadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/les/data"
lesname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_profiles.jld2"
filepath = joinpath(lesdatadir, lesname)

iters = iterations(filepath)
otherdata = ColumnData(filepath, reversed=true, initial=5, targets=[13, 121])
othermodel = ModularKPPOptimization.ColumnModel(otherdata, 5minute, Δ=2)

fig, axs = visualize_realizations(otherdata, othermodel,
                                  optimalparams, defaultparams;
                                  paramlabels=["optimal", "default"]
                                  )
gcf()
=#
