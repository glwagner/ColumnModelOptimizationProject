using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
#name = "mcmc_exp_batch_e1.0e-04_dt5.0_Δ2.jld2"
expname = "mcmc_exp_batch_s3.0e-05_dt5.0_Δ2.jld2"
regname = "mcmc_batch_s7.5e-05_dt5.0_Δ2.jld2"

expfilepath = joinpath(datadir, expname)
@load expfilepath chain
expchain = chain

regfilepath = joinpath(datadir, regname)
@load regfilepath chain
regchain = chain

i = 3

data = expchain.nll.batch[i].data
model = expchain.nll.batch[i].model

defaultparams = regchain[1].param
optimalexpparams = optimal(expchain).param
optimalregparams = optimal(regchain).param

legendkw = Dict(
    :markerscale=>1.2, :fontsize=>10,
    :loc=>"upper left", :bbox_to_anchor=>(0.2, 0.4),
    :frameon=>true, :framealpha=>1.0)

fig, axs = visualize_realizations(data, model,
                                 # optimalexpparams,
                                  optimalregparams,
                                  defaultparams,
                                  #paramlabels=["exponential modification", "standard KPP"],
                                  paramlabels=["optimal", "default"],
                                  i_data = [1, 9, 145]
                                  )
tight_layout()
gcf()
savefig("/Users/gregorywagner/Desktop/batch_optimization.png", dpi=480)

#=
d = 0:0.001:2
default(d, p) = d < 1 ? p * d * (1-d)^2 : 0.0
special(d, p) = default.(d, p.Cτ) .* (p.CS0 .+ p.CSe * exp.(-d/p.CSd))

fig, axs = subplots(figsize=(2, 4))
plot(default.(d, optimalregparams.Cτ), d, ":", linewidth=3)
plot(special(d, optimalexpparams), d, "-", linewidth=3)

axs.invert_yaxis()
axs.axis("off")
gcf()

savefig("/Users/gregorywagner/Desktop/kpp_shapes.png", dpi=480)
=#

#=
#
# Other data
#

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
