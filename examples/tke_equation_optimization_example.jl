using 
    OceanTurb,
    PyPlot,
    Dao,
    Distributions,
    ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization: ColumnModel, WindMixingParameters

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

# Model and data
data = ColumnData(datapath)
model = ColumnModel(data, 10second, N=32)

# Create loss function and negative-log-likelihood object
fields = (:U, :V, :T, :e)
#step = 50; targets = (step+1):step:401
targets = 21:401
loss = TimeAveragedLossFunction(data, targets=targets, fields=fields, weights=nothing)
nll = NegativeLogLikelihood(model, data, loss)

# Initial state for optimization step
default_parameters = DefaultFreeParameters(model, WindMixingParameters)
initial_covariance = Array([1e-2 for p in default_parameters])
bounds = [(0.0, 3.0) for p in default_parameters]

println("Optimizing...")
initial_iterations = 1000
covariance, chains = optimize(nll, default_parameters, initial_covariance,
                              BoundedNormalPerturbation, bounds,
                              samples = iter -> round(Int, initial_iterations/sqrt(iter)),
                              schedule = (nll, iter, link) -> exp(-iter+1) * link.error,
                              niterations=4)

@show optimal_link = optimal(chains[end])
optimal_parameters = optimal_link.param

visualize_realizations(model, data, [step+1, 401], default_parameters, optimal_parameters, 
                       fields=(:U, :V, :T, :e))

plot_loss_function(loss, model, data, default_parameters, optimal_parameters)
