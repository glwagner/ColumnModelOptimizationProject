using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

      N = 32                # Model resolution. Perfect model resolution is N=600
     dt = 10*minute         # Model timestep. Perfect model timestep is 1 minute.
   init = 2                # Choose initial condition for forward runs
targets = (10, 50, 90)    # Target samples of saved data for model-data comparison
   name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"

# Initialize the 'data' and the 'model'
 datadir = joinpath("les", "data")
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")
    data = ColumnData(filepath; initial=init, targets=targets, reversed=true)
   model = ModularKPPOptimization.ColumnModel(data, dt, N=N)

Uvariance = maxvariance(data, :U)
Vvariance = maxvariance(data, :V)
Tvariance = maxvariance(data, :T)

defaultparams = DefaultFreeParameters(model, WindMixingParameters)

println("We will optimize the following parameters:")
@show propertynames(defaultparams);

fig, axs = visualize_realization(defaultparams, model, data)
gcf()

nll = NegativeLogLikelihood(model, data, weighted_fields_loss,
    weights=(0.1 / Uvariance, 0.1 / Vvariance, 1.0 / Tvariance))

# Obtain the first link in the Markov chain
default_link = MarkovLink(nll, defaultparams)
error_ratio = 0.5
@show nll.scale = default_link.error * error_ratio

# Use a non-negative normal perturbation
relative_std = 0.02
stddev = WindMixingParameters(Tuple(relative_std for d in defaultparams)...)
bounds = WindMixingParameters(Tuple((0.0, max(10.0, 10d)) for d in defaultparams)...)
sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))

chainname = @sprintf("mcmc_%s_e%0.1e_std%0.1e_%03d", name, error_ratio,
                     relative_std, N)
chainpath = "$chainname.jld2"

dsave = 10^3
chain = MarkovChain(dsave, MarkovLink(nll, defaultparams), nll, sampler)
@show chain.acceptance
@save chainpath chain

#=
    tint = @elapsed extend!(chain, 10^3)
    @printf("First, optimal, and last links:\n")
    println((chain[1].error, chain[1].param))
    println((optimal(chain).error, optimal(chain).param))
    println((chain[end].error, chain[end].param))
    println(" ")
    println(status(chain))

    samples = params(chain)
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
=#

tstart = time()
while length(chain) < 10^7
    tint = @elapsed extend!(chain, dsave)

    @printf("tᵢ: %.2f seconds. Elapsed wall time: %.4f minutes.\n\n", tint, (time() - tstart)/60)
    @printf("First, optimal, and last links:\n")
    println((chain[1].error, chain[1].param))
    println((optimal(chain).error, optimal(chain).param))
    println((chain[end].error, chain[end].param))
    println(" ")

    println(status(chain))

    oldchainpath = chainname * "_old.jld2"
    mv(chainpath, oldchainpath, force=true)
    @save chainpath chain
    rm(oldchainpath)
end
