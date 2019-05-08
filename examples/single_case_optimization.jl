using Pkg; Pkg.activate("..")

using PyPlot, Printf, Statistics, OceanTurb, Dao, JLD2,
        ColumnModelOptimizationProject, ColumnModelOptimizationProject.KPPOptimization

model_N = 50
model_dt = 10*minute
initial_data = 1
target_data = (4, 7, 10)
cases = ("unstable_weak", "unstable_strong", "stable_weak", "stable_strong", "neutral")

datadir = joinpath("..", "data", "perfect_model")
filepaths = Dict((case, joinpath(datadir, case * ".jld2")) for case in cases)
datasets = Dict((case, ColumnData(filepaths[case]; initial=initial_data, targets=target_data)) for case in cases) 
models = Dict((case, KPPOptimization.ColumnModel(datasets[case], model_dt, N=model_N)) for case in cases) 

# Use "PriorityParameters"
defaults = DefaultFreeParameters(BasicParameters)

# Pick a case
case = "neutral"

model = models[case]
data = datasets[case]

# Obtain an estimate of the relative error in the temperature and velocity fields
test_nll_temperature = NegativeLogLikelihood(model, data, temperature_loss)
test_nll_velocity = NegativeLogLikelihood(model, data, velocity_loss)
test_link_temperature = MarkovLink(test_nll_temperature, defaults)
test_link_velocity = MarkovLink(test_nll_velocity, defaults)

error_ratio = test_link_velocity.error / test_link_temperature.error
@printf "Ratio of velocity error to temperature error: %.4f\n" error_ratio 

# Build the weighted NLL, normalizing temperature error relative to velocity error.
weights = (1, 1, 10*round(Int, error_ratio), 0)
nll = NegativeLogLikelihood(model, data, weighted_fields_loss, weights=weights) 

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, defaults)
nll.scale = first_link.error * 0.5

# Use a non-negative normal perturbation
std = DefaultStdFreeParameters(0.05, typeof(defaults))
sampler = MetropolisSampler(NormalPerturbation(std))

ninit = 10^3
chain = MarkovChain(ninit, first_link, nll, sampler)
@show chain.acceptance 

dsave = 10^3
chainname = case * "_markov_chain"
chainpath = "$chainname.jld2"
@save chainpath chain

tstart = time()
for i = 1:10
#while true
    tint = @elapsed extend!(chain, dsave)
    println(status(chain))

    @sprintf("Tc: %.2f seconds. Elapsed wall time: %.4f minutes.", tint, (time() - tstart)/60)
    println("Optimal parameters:")
    @show chain[1]
    @show optimal(chain)
    @show chain[end]

    oldchainpath = chainname * "_old.jld2"
    mv(chainpath, oldchainpath, force=true)
    @save chainpath chain
    rm(oldchainpath)
end
