using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

       N = 32 
      dt = 10*minute       
    init = 2              
 targets = (10, 30, 50)  
 r_error = 0.001
   r_std = 0.01
   Δsave = 10^3
  nlinks = 10^7
dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"

# Initialize the 'data' and the 'model'
 datadir = joinpath("les", "data")
filepath = joinpath(@__DIR__, "..", datadir, dataname * "_profiles.jld2")
    data = ColumnData(filepath; initial=init, targets=targets, reversed=true)
   model = ModularKPPOptimization.ColumnModel(data, dt, N=N)

chainname = @sprintf("mcmc_shape_%s_e%0.1e_std%0.1e_%03d", name, r_error, r_std, N)
chainpath(name) = joinpath(@__DIR__, "data", "$name.jld2")

nll = NegativeLogLikelihood(model, data, weighted_fields_loss,
                            weights = Tuple(1/maxvariance(data, fld) for fld in (:U, :V, :T))
                           )

# Set up the Markov Chain
defaultparams = DefaultFreeParameters(model, WindMixingAndShapeParameters)

# Obtain the first link in the Markov chain
defaultlink = MarkovLink(nll, defaultparams)

# Use a non-negative normal perturbation
stddev = WindMixingAndShapeParameters(
                              0.001,
                              0.005,
                              0.001,
                              0.0001,
                              0.0001,
                             )

bounds = WindMixingAndShapeParameters(
                              (0.0, 1.0),
                              (0.0, 1.0),
                              (0.0, 2.0),
                              (0.0, 1.0),
                              (-1.0, 2.0),
                             )

sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))
chain = MarkovChain(Δsave, MarkovLink(nll, defaultparams), nll, sampler)

tstart = time()
while length(chain) < nlinks
    tint = @elapsed extend!(chain, Δsave)

    @printf("tᵢ: %.2f seconds. Elapsed wall time: %.4f minutes.\n\n", tint, (time() - tstart)/60)
    @printf("First, optimal, and last links:\n")
    println((chain[1].error, chain[1].param))
    println((optimal(chain).error, optimal(chain).param))
    println((chain[end].error, chain[end].param))
    println(" ")

    println(status(chain))

    # Save results in conservative manner
    oldchainpath = chainpath(name * "_old")
    newchainpath = chainpath(name)
    mv(newchainpath, oldchainpath, force=true)
    @save newchainpath chain
    rm(oldchainpath)
end
