using Pkg; Pkg.activate("..")

using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2,
        ColumnModelOptimizationProject, ColumnModelOptimizationProject.KPPOptimization

     model_N = 30                # Model resolution.
    model_dt = 10*minute         # Model timestep.
initial_data = 1                 # Choose initial condition for forward runs
 target_data = (4, 7, 10)        # Target samples of saved data for model-data comparison

cases = ("free_convection", "unstable_weak", "unstable_strong", "stable_weak", "stable_strong", "neutral")

  datadir = joinpath("..", "data", "coarse_perfect_model_experiment")
filepaths = Dict((case, joinpath(datadir, case * ".jld2")) for case in cases)
 datasets = Dict((case, ColumnData(filepaths[case]; initial=initial_data, targets=target_data)) for case in cases) 
   models = Dict((case, KPPOptimization.ColumnModel(datasets[case], model_dt, N=model_N)) for case in cases) 

# Pick a set of parameters to optimize
defaults = DefaultFreeParameters(BasicParameters)
std = DefaultStdFreeParameters(0.05, typeof(defaults))
perturbed = NormalPerturbation(std)(defaults)

#=
# Obtain an estimate of the relative size of velocity versus temperature error
ratios = Dict()
for case in cases
    model = models[case]
    data = datasets[case]

     test_nll_temperature = NegativeLogLikelihood(model, data, temperature_loss)
        test_nll_velocity = NegativeLogLikelihood(model, data, velocity_loss)
    test_link_temperature = MarkovLink(test_nll_temperature, perturbed)
       test_link_velocity = MarkovLink(test_nll_velocity, perturbed)

     error_ratio = test_link_velocity.error / test_link_temperature.error
    ratios[case] = error_ratio
end

# Normalize temperature error relative to velocity error
@show weights = (1, 1, 10*round(Int, mean(values(ratios))/10), 0)
=#
weights = (1, 1, 10, 0)

# Build the batch of NLLs.
nll = BatchedNegativeLogLikelihood(
    [ NegativeLogLikelihood(models[case], datasets[case], weighted_fields_loss, weights=weights) 
        for case in cases ])

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, perturbed)

@show first_link.error
nll.scale = first_link.error * 1.0

std = DefaultStdFreeParameters(0.05, typeof(defaults))
bounds = BasicParameters(((0.0*p, 3.0*p) for p in defaults)...)
sampler = MetropolisSampler(BoundedNormalPerturbation(std, bounds))

dsave = 10^4
chain = MarkovChain(dsave, first_link, nll, sampler)
@show chain.acceptance 

chainname = @sprintf("coarse_perfect_model_batch_markov_N%03d", model_N)
chainpath = "$chainname.jld2"

isfile(chainpath) && rm(chainpath)

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
