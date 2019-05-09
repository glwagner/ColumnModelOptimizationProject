using Pkg; Pkg.activate("..")

using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2,
        ColumnModelOptimizationProject, ColumnModelOptimizationProject.KPPOptimization

     model_N = 30                # Model resolution. Perfect model resolution is N=600
    model_dt = 10*minute         # Model timestep. Perfect model timestep is 1 minute.
initial_data = 1                 # Choose initial condition for forward runs
 target_data = (4, 7, 10)        # Target samples of saved data for model-data comparison
        case = "unstable_weak"   # Case to run comparison

# Initialize the 'data' and the 'model'
 datadir = joinpath("..", "data", "perfect_model_experiment")
filepath = joinpath(datadir, case * ".jld2")
    data = ColumnData(filepath; initial=initial_data, targets=target_data)
   model = KPPOptimization.ColumnModel(data, model_dt, N=model_N)

# Pick a set of parameters to optimize
defaults = DefaultFreeParameters(BasicParameters)

println("We will optimize the following parameters:")
@show propertynames(defaults);

# Obtain an estimate of the relative error in the temperature and velocity fields
 test_nll_temperature = NegativeLogLikelihood(model, data, temperature_loss)
    test_nll_velocity = NegativeLogLikelihood(model, data, velocity_loss)
test_link_temperature = MarkovLink(test_nll_temperature, defaults)
   test_link_velocity = MarkovLink(test_nll_velocity, defaults)

@show error_ratio = test_link_velocity.error / test_link_temperature.error

# Build the weighted NLL, normalizing temperature error relative to velocity error.
weights = (1, 1, 10*round(Int, error_ratio/10), 0)
nll = NegativeLogLikelihood(model, data, weighted_fields_loss, weights=weights) 

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, defaults)
nll.scale = first_link.error * 0.5

# Use a non-negative normal perturbation
std = DefaultStdFreeParameters(0.05, typeof(defaults))
sampler = MetropolisSampler(NormalPerturbation(std))

dsave = 10^4
chain = MarkovChain(dsave, first_link, nll, sampler)
@show chain.acceptance 

chainname = @sprintf("perfect_model_%s_markov_N%03d", case, model_N)
chainpath = "$chainname.jld2"
@save chainpath chain

tstart = time()
while length(chain) < 10^6
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
