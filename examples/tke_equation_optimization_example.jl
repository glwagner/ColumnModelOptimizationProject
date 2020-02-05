using 
    OceanTurb,
    PyPlot,
    Dao,
    Distributions,
    ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization: WindMixingParameters

#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
datapath = "stress_driven_Nsq1.6e-05_f0.0e+00_Qu1.0e-04_Nh128_Nz128_averages.jld2"

# Model and data
data = ColumnData(datapath)
model = TKEMassFluxOptimization.ColumnModel(data, 1minute, N=32)

# Create loss function and negative-log-likelihood object
targets = 801:21:1601
fields = (:T, :e)
max_variances = [max_variance(data, field, targets) for field in fields]
@show max_variances ./= maximum(max_variances)
weights = (1, 10 * max_variances[2])
loss = TimeAveragedLossFunction(data, targets=targets, fields=fields, weights=weights)
nll = NegativeLogLikelihood(model, data, loss)

# Initial state for optimization step
default_parameters = DefaultFreeParameters(model, WindMixingParameters)
initial_covariance = Array([1e-3 for p in default_parameters])
bounds = [(0.0, 3.0) for p in default_parameters]

println("Optimizing...")
initial_iterations = 100
covariance, chains = optimize(nll, default_parameters, initial_covariance,
                              BoundedNormalPerturbation, bounds,
                              samples = iter -> round(Int, initial_iterations/sqrt(iter)),
                              schedule = (nll, iter, link) -> link.error / iter^4,
                              niterations=4)

@show optimal_link = optimal(chains[end])
optimal_parameters = optimal_link.param

visualize_realizations(model, data, loss.targets[[1, end]], default_parameters, optimal_parameters, 
                       fields=(:U, :V, :T, :e))

plot_loss_function(loss, model, data, default_parameters, optimal_parameters)
visualize_loss_function(loss, model, data, loss.targets[end], default_parameters, optimal_parameters)
