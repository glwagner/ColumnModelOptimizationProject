using Printf, Statistics, OceanTurb, Dao, JLD2, OffsetArrays, LinearAlgebra

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

     model_N = 64                # Model resolution. Perfect model resolution is N=600
    model_dt = 10*minute         # Model timestep. Perfect model timestep is 1 minute.
initial_data = 2                # Choose initial condition for forward runs
 target_data = (10, 50, 90)    # Target samples of saved data for model-data comparison
        name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
 mixingdepth = ModularKPP.LMDMixingDepth()
 mixingdepthname = "CVMix"

#mixingdepth = ModularKPP.ROMSMixingDepth()
#mixingdepthname = "ROMS"

# Initialize the 'data' and the 'model'
 datadir = joinpath("les", "data")
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")
    data = ColumnData(filepath; initial=initial_data, targets=target_data)
   model = ModularKPPOptimization.ColumnModel(data, model_dt, N=model_N,
                                              mixingdepth=mixingdepth)

Uvariance = maxvariance(data, :U)
Vvariance = maxvariance(data, :V)
Tvariance = maxvariance(data, :T)

# Pick a set of parameters to optimize
defaults = DefaultFreeParameters(model, WindMixingParameters)

println("We will optimize the following parameters:")
@show propertynames(defaults);

nll = NegativeLogLikelihood(model, data, weighted_fields_loss,
    weights=(0.1 / Uvariance, 0.1 / Vvariance, 1.0 / Tvariance))

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, defaults)
@show nll.scale = first_link.error * 0.1 #1e-2

# Use a non-negative normal perturbation
stddev = WindMixingParameters(Tuple(0.05 for d in defaults)...)
bounds = WindMixingParameters(Tuple((0.0, max(5.0, 5d)) for d in defaults)...)
sampler = MetropolisSampler(BoundedNormalPerturbation(stddev, bounds))

chainname = @sprintf("%s_%s_markov_N%03d_2", name, mixingdepthname, model_N)
chainpath = "$chainname.jld2"

dsave = 10^2
chain = MarkovChain(dsave, first_link, nll, sampler)
@show chain.acceptance

@save chainpath chain

#@load chainpath chain

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
