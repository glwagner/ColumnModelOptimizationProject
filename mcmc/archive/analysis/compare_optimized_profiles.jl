using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
#name = "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_e1.0e-03_dt5.0_Δ2.jld2"
name = "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2"
#name = "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2"

filepath = joinpath(datadir, name)
@load filepath chain

data = chain.nll.data
model = chain.nll.model
defaultparams = chain[1].param
optimalparams = optimal(chain).param

legendkw = Dict(
    :markerscale=>1.2, :fontsize=>10,
    :loc=>"bottom right", :bbox_to_anchor=>(0.2, 0.7),
    :frameon=>true, :framealpha=>1.0)

fig, axs = visualize_realizations(data, model,
                                  optimalparams, defaultparams;
                                  paramlabels=["optimal", "default"],
                                  i_data = [5, 9, 121],
                                  legendkwargs = legendkw
                                  )

tight_layout()
gcf()

savefig("/Users/gregorywagner/Desktop/example_optimization.png", dpi=480)

#=
fig, axs = subplots()
samples = Dao.params(chain)
CRi = map(x->getproperty(x, :CRi), samples)
Cτ = map(x->getproperty(x, :Cτ), samples)
CSL = map(x->getproperty(x, :CSL), samples)

scatter(CRi, Cτ, alpha=0.1)
plot(optimalparams.CRi, optimalparams.Cτ, "b*", markersize=10)
xlabel(L"C^\mathrm{Ri}")
ylabel(L"C^\tau")
OceanTurbPyPlotUtils.removespines("top", "right")
gcf()

savefig("/Users/gregorywagner/Desktop/example_mcmc.png", dpi=480)
=#


#
# Other data
#

#=
datadir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/les/data"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz128_Nz256"
filepath = joinpath(datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=5, targets=[529])
model = ModularKPPOptimization.ColumnModel(data, 5minute, Δ=2)

#fig, axs = visualize_realization(optimalparams, model, data; legendkwargs=legendkw)
#tight_layout()
#gcf()

fig, axs = visualize_realizations(data, model,
                                  optimalparams, defaultparams;
                                  paramlabels=["optimal", "default"],
                                  )
tight_layout()
gcf()
=#
