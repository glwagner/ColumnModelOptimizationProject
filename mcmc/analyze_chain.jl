using Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

chainname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_CVMix_tinyscale_markov_N064.jld2"
chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc"
chainpath = joinpath(chaindir, chainname)

@load chainpath chain

length(chain)
samples = Dao.params(chain)

opt = optimal(chain)

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
