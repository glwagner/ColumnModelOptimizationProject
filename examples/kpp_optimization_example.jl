using Pkg; Pkg.activate("..")

using PyPlot, Printf, Statistics, OceanTurb, Dao, 
        ColumnModelOptimizationProject, ColumnModelOptimizationProject.KPPOptimization

model_N = 50
model_dt = 10*minute
initial_data = 1
target_data = (4, 7, 10)
cases = ("unstable_weak", "unstable_strong", "stable_weak", "stable_strong", "neutral")

datadir = joinpath("..", "data", "perfect_model")
filepaths = Dict((case, joinpath(datadir, case * ".jld2")) for case in cases)
datasets = Dict(
    (case, ColumnData(filepaths[case]; initial=initial_data, targets=target_data)) for case in cases) 
models = Dict(
    (case, KPPOptimization.ColumnModel(datasets[case], model_dt, N=model_N)) for case in cases) 

Base.@kwdef mutable struct InterestingParameters{T} <: FreeParameters{9, T}
    CRi   :: T # Critical bulk Richardson number
    CKE   :: T # Unresolved kinetic energy constant
    CNL   :: T # Non-local flux constant
    Cτ    :: T # Von Karman constant
    Cunst :: T # Unstable buoyancy flux parameter for wind-driven turbulence
    Cb_U  :: T # Buoyancy flux parameter for convective turbulence
    Cb_T  :: T # Buoyancy flux parameter for convective turbulence
    Cd_U  :: T # Wind mixing regime threshold for momentum
    Cd_T  :: T # Wind mixing regime threshold for tracers
end

defaults = DefaultFreeParameters(InterestingParameters)
ratios = Dict()

# Obtain an estimate of the relative size of velocity versus temperature error
for case in cases
    model = models[case]
    data = datasets[case]

    test_nll_temperature = NegativeLogLikelihood(model, data, temperature_loss)
    test_nll_velocity = NegativeLogLikelihood(model, data, velocity_loss)

    test_link_temperature = MarkovLink(test_nll_temperature, defaults)
    test_link_velocity = MarkovLink(test_nll_velocity, defaults)

    error_ratio = test_link_velocity.error / test_link_temperature.error
    ratios[case] = error_ratio
end

@show "ratio of velocity error to temperature error" ratios

# Normalize temperature error relative to velocity error
weights = (1, 1, 10*round(Int, mean(values(ratios))/10), 0)

# Build the batch of NLLs.
nll = BatchedNegativeLogLikelihood(
    [ NegativeLogLikelihood(models[case], datasets[case], weighted_fields_loss, weights=weights) 
        for case in cases ])

# Obtain the first link in the Markov chain
first_link = MarkovLink(nll, defaults)

@show first_link.error
nll.scale = first_link.error * 0.1

std = DefaultStdFreeParameters(0.1, typeof(defaults))
sampler = MetropolisSampler(NonNegativeNormalPerturbation(std))

n = 100
tn = @elapsed chain = MarkovChain(n, first_link, nll, sampler)

@show chain.acceptance 

@printf "time per link: %.6f s" tn/n
