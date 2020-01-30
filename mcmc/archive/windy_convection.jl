using 
    PyPlot, Printf, JLD2, OceanTurb,
    Dao, ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

       Δ = 2        # Coarse model with 4m resolution
      dt = 10minute # 10 minute time-steps
 r_error = 0.001
   Δsave = 10^3
dataname ="simple_flux_Fb1e-08_Fu-1e-04_Nsq2e-06_Lz64_Nz256"
savename = @sprintf("mcmc_%s_e%0.1e_%d", dataname, r_error, Δ)
savepath(name) = name * ".jld2"

# Initialize the 'data' and the 'model'
datapath = joinpath(@__DIR__, "..", "les", "data", dataname * "_profiles.jld2")
    data = ColumnData(datapath; initial=2, targets=(10, 30), reversed=true)
   model = ModularKPPOptimization.ColumnModel(data, dt, Δ=Δ)

# Set up a Negative Log Likelihood function using the maximum
# measure variance as weighting
weights = Tuple(1/maxvariance(data, fld) for fld in (:U, :V, :T))
nll = NegativeLogLikelihood(model, data, weighted_fields_loss, weights=weights)

# Set up the Markov Chain, using error associated with 
# default parameters to determine the loss function scale/temperature
defaultparams = DefaultFreeParameters(model, WindyConvectionParameters)
defaultlink = MarkovLink(nll, defaultparams)
nll.scale = r_error * defaultlink.error

# Set up a random talk on periodic domain.
stddev = WindyConvectionParameters((5e-3 for p in defaultparams)...)
bounds = WindyConvectionParameters(((0.0, 20.0) for p in defaultparams)...)

sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))
chain = MarkovChain(Δsave, MarkovLink(nll, defaultparams), nll, sampler)
@save savepath(savename) chain

tstart = time()
while length(chain) < 10^5
    tint = @elapsed extend!(chain, Δsave)

    @printf("Elapsed wall time: %.4f minutes (Δ: %.1f s).\n\n", 
            (time() - tstart)/60, tint)

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
    @save newsavepath chain
    rm(oldsavepath)
end
