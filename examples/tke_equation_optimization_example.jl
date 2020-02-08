using 
    OceanTurb,
    PyPlot,
    Dao,
    Distributions,
    ColumnModelOptimizationProject

using Statistics: mean

using ColumnModelOptimizationProject.TKEMassFluxOptimization

#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
datapath = "stress_driven_Nsq1.6e-05_f0.0e+00_Qu1.0e-04_Nh128_Nz128_averages.jld2"

# Model and data
data = ColumnData(datapath)
model = TKEMassFluxOptimization.ColumnModel(data, 1minute, N=64,
                                            mixing_length=TKEMassFlux.SimpleMixingLength())

# Create loss function and negative-log-likelihood object
targets = 51:10:401
fields = (:T, :U)

max_variances = [max_variance(data, field) for field in fields]
weights = [1/σ for σ in max_variances]
weights[1] *= 100

loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights)
nll = NegativeLogLikelihood(model, data, loss)

@free_parameters TestParameters Cᴷᵤ Cᴷₑ Cᴾʳ

#=
Base.@kwdef mutable struct TestParameters{T} <: FreeParameters{3, T}
     Cᴷᵤ :: T
     Cᴷₑ :: T
     Cᴾʳ :: T
end
=#

# Initial state for optimization step
ParameterSet = TestParameters
default_parameters = DefaultFreeParameters(model, ParameterSet)

initial_variance = ParameterSet((1e-1 for p in default_parameters)...)
initial_variance.Cᴷᵤ = 1e-2
initial_variance = Array(initial_variance)

bounds = ParameterSet(((0.0, 3.0) for p in default_parameters)...)
bounds.Cᴷᵤ = (0.1, 1.0) # prevent optimum near 0

# Initialize
initial_chain = MarkovChain(1000, MarkovLink(nll, default_parameters), nll,
                            MetropolisSampler(BoundedNormalPerturbation(initial_variance, bounds)))

# Optimization step
nsamples = 100
niters = 4
covariance, chains = optimize(nll, optimal(initial_chain).param, 
                              initial_variance, BoundedNormalPerturbation, bounds;
                              # keyword args:
                                 samples = nsamples,
                                schedule = (nll, iter, link) -> link.error / iter,
                              iterations = niters)

@show optimal_link = optimal(chains[end])
optimal_parameters = optimal_link.param

visualize_realizations(model, data, loss.targets[[1, end]], default_parameters, optimal_parameters, 
                       fields=(:U, :T, :e))

plot_loss_function(loss, model, data, default_parameters, optimal_parameters, time_norm=:hour)
visualize_loss_function(loss, model, data, length(loss.targets), default_parameters, optimal_parameters)

chain = chains[end]

parameters_to_plot = propertynames(default_parameters) #(:Cᴰ, :Cᴷᵤ, :Cᴷₑ, :Cᴾʳ, :Cʷu★)
fig, axs = subplots(nrows=length(parameters_to_plot), figsize=(5, 8))

for i = 1:3
    extend!(chain, 1000)

    for ax in axs
        sca(ax); cla()
    end

    for (i, parameter) in enumerate(parameters_to_plot)
        visualize_markov_chain!(axs[i], chain, parameter)
        sca(axs[i])
        xlabel(parameter_latex_guide[parameter])
        tight_layout()
    end
end
