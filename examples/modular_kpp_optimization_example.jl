using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)
model = ModularKPPOptimization.ColumnModel(data, 5minute, N=64)

loss_function = TimeAveragedLossFunction(targets=1:length(data), fields=:T)

nll = NegativeLogLikelihood(model, data, loss_function)
nll.scale = 1e3

default_parameters = DefaultFreeParameters(model, WindMixingParameters)
default_link = MarkovLink(nll, default_parameters)

standard_deviation = WindMixingParameters((1e-2 for p in default_parameters)...)
bounds = WindMixingParameters((0.0, 2.0), (0.0, 1.0), (0.0, 2.0))

sampler = MetropolisSampler(BoundedNormalPerturbation(standard_deviation, bounds))
chain = MarkovChain(100, MarkovLink(nll, default_parameters), nll, sampler)

@show chain.acceptance
@show optimal(chain)

#visualize_realization(optimal(chain).param, model, data, loss_function.targets[[1, 201, 401]])
visualize_realizations(model, data, loss_function.targets[[1, 401]], chain[1].param, optimal(chain).param)

#fig, axs = summarize_data(datapath)


