using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

#chainname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_CVMix_markov_N016.jld2"
#chainname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_CVMix_markov_N064_2.jld2"
#chainname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_CVMix_markov_bigscale_N032.jld2"
#chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e+00_std1.0e-02_032.jld2"
chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e5.0e-01_std2.0e-02_032.jld2"
chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc"
chainpath = joinpath(chaindir, chainname)

@load chainpath chain

length(chain)
samples = Dao.params(chain)

opt = optimal(chain)
@show opt.param

CRi = map(x->x.CRi, samples)
 Cτ = map(x->x.Cτ, samples)
CSL = map(x->x.CSL, samples)

fig, axs = subplots(nrows=3)

sca(axs[1])
plt.hist(CRi, bins=100, density=true)
plt.plot(opt.param.CRi, 1, "k*", markeredgecolor=nothing)

sca(axs[2])
plt.hist(Cτ, bins=100, density=true)
plt.plot(opt.param.Cτ, 1, "k*", markeredgecolor=nothing)

sca(axs[3])
plt.hist(CSL, bins=100, density=true)
plt.plot(opt.param.CSL, 1, "k*", markeredgecolor=nothing)

tight_layout()

gcf()

fig, axs = visualize_realization(WindMixingParameters(opt.param...), chain.nll.model, chain.nll.data)
gcf()
