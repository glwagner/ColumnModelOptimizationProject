using 
    OceanTurb,
    PyPlot,
    Dao,
    Distributions,
    ColumnModelOptimizationProject

using Statistics: mean

using ColumnModelOptimizationProject.TKEMassFluxOptimization: WindMixingParameters

#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
datapath = "stress_driven_Nsq1.6e-05_f0.0e+00_Qu1.0e-04_Nh128_Nz128_averages.jld2"

# Model and data
data = ColumnData(datapath)
model = TKEMassFluxOptimization.ColumnModel(data, 1minute, N=32)

# Create loss function and negative-log-likelihood object
targets = 11:401
fields = (:T, :U, :V, :e)

max_variances = [max_variance(data, field) for field in fields]
weights = [1/σ for σ in max_variances]
weights[1] *= 10
weights[4] /= 2

loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights)
nll = NegativeLogLikelihood(model, data, loss)

# Initial state for optimization step
default_parameters = DefaultFreeParameters(model, WindMixingParameters)
initial_covariance = Array([1e-1 for p in default_parameters])
bounds = [(0.0, 3.0) for p in default_parameters]

samples = 4000
covariance, chains = optimize(nll, default_parameters, initial_covariance,
                              BoundedNormalPerturbation, bounds,
                              samples = samples,
                              schedule = (nll, iter, link) -> 4 * link.error / iter,
                              niterations=3)

@show optimal_link = optimal(chains[end])
optimal_parameters = optimal_link.param

visualize_realizations(model, data, loss.targets[[1, end]], default_parameters, optimal_parameters, 
                       fields=(:U, :V, :T, :e))

plot_loss_function(loss, model, data, default_parameters, optimal_parameters, time_norm=:hour)
visualize_loss_function(loss, model, data, length(loss.targets), default_parameters, optimal_parameters)

chain = chains[end]
extend!(chain, samples)

fig, axs = subplots()

visualize_markov_chain!(axs[1], chain, :Cᴰ)
visualize_markov_chain!(axs[2], chain, :Cᴷ)
visualize_markov_chain!(axs[2], chain, :Cᴾʳ)
visualize_markov_chain!(axs[3], chain, :Cʷu★)
