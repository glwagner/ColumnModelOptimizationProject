using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

       Δ = 2        # Model resolution
      dt = 5minute  # 10 minute time-steps
 r_error = 0.002
   Δsave = 10^2

#dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz64_Nz128"
#dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
#dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128"
dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-06_Lz64_Nz128"

savename = @sprintf("mcmc_strat_%s_e%0.1e_dt%.1f_Δ%d", dataname, r_error, dt/minute, Δ)
savepath(name) = joinpath("data", name * ".jld2")

# Initialize the 'data' and the 'model'
datapath = joinpath(@__DIR__, "..", "les", "data", dataname * "_profiles.jld2")
    #data = ColumnData(datapath; initial=5, targets=[9, 25, 121], reversed=true)
    data = ColumnData(datapath; initial=5, targets=[9, 25], reversed=true)
   model = ModularKPPOptimization.ColumnModel(data, dt, Δ=Δ)

# Set up a Negative Log Likelihood function using the maximum
# measure variance as weighting
weights = Tuple(1/maxvariance(data, fld) for fld in (:U, :V, :T))
nll = NegativeLogLikelihood(model, data, weighted_fields_loss, weights=weights)

# Set up the Markov Chain, using error associated with 
# default parameters to determine the loss function scale/temperature
defaultparams = DefaultFreeParameters(model, WindMixingParameters)
defaultlink = MarkovLink(nll, defaultparams)
nll.scale = r_error * defaultlink.error

# Set up a random talk on periodic domain.
stddev = WindMixingParameters((5e-3 for p in defaultparams)...)
bounds = WindMixingParameters(
                              (0.0, 2.0),
                              (0.0, 2.0),
                              (0.0, 2.0)
                             )

sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))
chain = MarkovChain(Δsave, MarkovLink(nll, defaultparams), nll, sampler)
@save savepath(savename) chain

tstart = time()
while length(chain) < 10^7
    tint = @elapsed extend!(chain, Δsave)

    @printf("Elapsed wall time: %.4f minutes (Δ: %.1f s).\n\n", (time() - tstart)/60, tint)
    @printf("Markov chain with parameters %s:\n", paramnames(chain))
    @printf("  first: %.4f, %s\n", chain[1].error,       "$(chain[1].param)")
    @printf("optimal: %.4f, %s\n", optimal(chain).error, "$(optimal(chain).param)")
    @printf("   last: %.4f, %s\n", chain[end].error,     "$(chain[end].param)")

    println("")
    println(status(chain))

    # Save results in conservative manner
    oldsavepath = savepath(savename * "_old")
    newsavepath = savepath(savename)
    mv(newsavepath, oldsavepath, force=true)

    println("Saving Markov chain data to $newsavepath.")
    @save newsavepath chain

    rm(oldsavepath)
end
