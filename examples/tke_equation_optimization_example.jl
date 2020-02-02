using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.TKEMassFluxOptimization

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)

model = TKEMassFluxOptimization.ColumnModel(data, 5minute, N=64)

default_parameters = DefaultFreeParameters(model, WindMixingParameters)
standard_deviation = Array(1e-1 * default_parameters)
bounds = [(0.0, 2.0), (0.0, 1.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0)]

loss_function = TimeAveragedLossFunction(targets=11:10:201, fields=:T)

nll = NegativeLogLikelihood(model, data, loss_function)

# Initialize sampler and NLL
nll.scale = 1.0
default_link = MarkovLink(nll, default_parameters)

iterations = 10
initial_steps = 25
sampler = MetropolisSampler(BoundedNormalPerturbation(standard_deviation, bounds))
initial_link = MarkovLink(nll, default_parameters)
@time chain = MarkovChain(100, initial_link, nll, sampler)
visualize_realizations(model, data, loss_function.targets[[1, length(loss_function.targets)]], 
                       chain[1].param, optimal(chain).param)
