using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)

model = ModularKPPOptimization.ColumnModel(data, 2minute, N=48)
default_parameters = DefaultFreeParameters(model, WindMixingParameters)
standard_deviation = [1e-1 for i = 1:length(default_parameters)]
bounds = [(0.0, 3.0), (0.0, 1.0), (0.0, 3.0)]

loss_function = TimeAveragedLossFunction(data, targets=11:10:length(data), 
                                         fields=(:U, :V, :T), weights=(1, 0.1, 1.0))

nll = NegativeLogLikelihood(model, data, loss_function)

# Initialize sampler and NLL
niterations = 3
schedule(nll, iteration, initial_link) = 1 / iteration^2 * initial_link.error
samples(iteration) = round(Int, 10000 / iteration^2)

covariance, chains = optimize(nll, default_parameters, standard_deviation, 
                             BoundedNormalPerturbation, bounds, samples=samples, 
                             schedule=schedule, niterations=niterations)


for chain in chains
    @show chain
end

@show optimal_link = optimal(chains[end])

visualize_realizations(model, data, [1, 201, 401],
                       default_parameters, optimal_link.param)

evaluate!(loss_function, default_parameters, model, data)

fig, axs = subplots()
plot(loss_function.error)

evaluate!(loss_function, optimal_link.param, model, data)

plot(loss_function.error)
