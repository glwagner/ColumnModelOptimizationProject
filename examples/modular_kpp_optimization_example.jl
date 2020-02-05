using 
    OceanTurb,
    PyPlot,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization

function plot_two_errors(loss, model, data, param1, param2; labels=["Default", "Optimal"])
    fig, axs = subplots()

    evaluate!(loss, param1, model, data)
    plot(loss.time, loss.error, label=labels[1])

    evaluate!(loss, param2, model, data)
    plot(loss.time, loss.error, label=labels[2])

    return fig, axs
end

datapath = "stress_driven_Nsq1.6e-05_f0.0e+00_Qu1.0e-04_Nh128_Nz128_averages.jld2"
#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)
model = ModularKPPOptimization.ColumnModel(data, 5minute, N=32)

fields = (:U, :T)
targets = 101:100:1601
loss_function = TimeAveragedLossFunction(data, targets=targets, fields=fields, weights=nothing)
nll = NegativeLogLikelihood(model, data, loss_function)

default_parameters = DefaultFreeParameters(model, WindMixingParameters)
standard_deviation = [1e-3 for i = 1:length(default_parameters)]
bounds = [(0.0, 3.0), (0.0, 1.0), (0.0, 3.0)]
covariance, chains = optimize(nll, default_parameters, standard_deviation,
                              BoundedNormalPerturbation, bounds, 
                               samples = iter -> round(Int, 100/sqrt(iter)), 
                              schedule = (nll, iter, link) -> exp(-iter+1) * link.error,
                              niterations=4)

@show optimal_link = optimal(chains[end])
optimal_parameters = optimal_link.param
visualize_realizations(model, data, [101, 1601], default_parameters, optimal_parameters)

fig, axs = plot_two_errors(loss_function, model, data, default_parameters, optimal_parameters)
