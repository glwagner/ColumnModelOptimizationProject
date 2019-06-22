using 
    PyPlot, Printf, JLD2,
    Dao, ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

       Δ = 4       # Coarse model with 4m resolution
      dt = 10 * 60 # 10 minute time-steps
 r_error = 0.001
   Δsave = 10^3
dataname = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
savename = "kpp_optimization_example"

savepath(name) = name * ".jld2"

# Initialize the 'data' and the 'model'
datapath = joinpath(@__DIR__, "..", "data", dataname * "_profiles.jld2")
    data = ColumnData(datapath; initial=3, targets=(13, 97), reversed=true)
   model = ModularKPPOptimization.ColumnModel(data, dt, Δ=Δ)

# Set up a Negative Log Likelihood function using the maximum
# measure variance as weighting
weights = Tuple(1/maxvariance(data, fld) for fld in (:U, :V, :T))
nll = NegativeLogLikelihood(model, data, weighted_fields_loss, weights=weights)

# Set up the Markov Chain, using error associated with 
# default parameters to determine the loss function scale/temperature
defaultparams = DefaultFreeParameters(model, WindMixingParameters)
defaultlink = MarkovLink(nll, defaultparams)
nll.scale = 1e-3 * defaultlink.error

# Set up a random talk on periodic domain.
stddev = WindMixingParameters(0.005, 0.005, 0.005)
bounds = WindMixingParameters(
                              (0.0, 1.0), # Bulk Richardson number bounds
                              (0.0, 1.0), # Surface layer fraction bounds
                              (0.0, 2.0)  # Diffusivity magnitude bounds
                             )

sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))
chain = MarkovChain(Δsave, MarkovLink(nll, defaultparams), nll, sampler)
@save savepath(savename) chain

tstart = time()
while length(chain) < 10^5
    tint = @elapsed extend!(chain, Δsave)

    @printf("tᵢ: %.2f seconds. Elapsed wall time: %.4f minutes.\n\n", tint, (time() - tstart)/60)
    @printf("First, optimal, and last links:\n")
    println((chain[1].error, chain[1].param))
    println((optimal(chain).error, optimal(chain).param))
    println((chain[end].error, chain[end].param))
    println(" ")

    println(status(chain))

    # Save results in conservative manner
    oldsavepath = savepath(chainname * "_old")
    newsavepath = savepath(chainname)
    mv(newsavepath, oldsavepath, force=true)
    @save newsavepath chain
    rm(oldsavepath)
end
