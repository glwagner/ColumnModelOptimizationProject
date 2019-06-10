using Printf, Statistics, OceanTurb, Dao, JLD2

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

     model_N = 128               # Model resolution. Perfect model resolution is N=600
    model_dt = 10*minute         # Model timestep. Perfect model timestep is 1 minute.
initial_data = 10                # Choose initial condition for forward runs
 target_data = (50, 100, 150)    # Target samples of saved data for model-data comparison
        name = "simple_flux_Fb5e-09_Fu-1e-04_Nsq1e-06_Lz128_Nz256"
 #mixingdepth = ModularKPP.LMDMixingDepth()
 #mixingdepthname = "CVMix"

mixingdepth = ModularKPP.ROMSMixingDepth()
mixingdepthname = "ROMS"

# Initialize the 'data' and the 'model'
 datadir = "data"
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")
    data = ColumnData(filepath; initial=initial_data, targets=target_data)
   model = ModularKPPOptimization.ColumnModel(data, model_dt, N=model_N,
                                              mixingdepth=mixingdepth)

# Pick a set of parameters to optimize
defaults = DefaultFreeParameters(model, BasicParameters)

println("We will optimize the following parameters:")
@show propertynames(defaults);

# Obtain an estimate of the relative error in the temperature and velocity fields
 test_nll_temperature = NegativeLogLikelihood(model, data, temperature_loss)
test_link_temperature = MarkovLink(test_nll_temperature, defaults)

nll = NegativeLogLikelihood(model, data, temperature_loss)

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, defaults)
nll.scale = first_link.error * 0.1

# Use a non-negative normal perturbation
stddev = DefaultStdFreeParameters(0.05, typeof(defaults))
sampler = MetropolisSampler(NormalPerturbation(stddev))

dsave = 10^2
chain = MarkovChain(dsave, first_link, nll, sampler)
@show chain.acceptance

chainname = @sprintf("%s_%s_markov_N%03d", name, mixingdepthname, model_N)
chainpath = "$chainname.jld2"
@save chainpath chain

tstart = time()
while length(chain) < 3 * 10^2
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
