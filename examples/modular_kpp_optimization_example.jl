using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)

model = ModularKPPOptimization.ColumnModel(data, 5minute, N=64, 
                                           mixingdepth = ModularKPP.ROMSMixingDepth(),
                                              kprofile = ModularKPP.GeneralizedCubicPolynomial()
)

default_parameters = DefaultFreeParameters(model, WindMixingAndShapeParameters)
standard_deviation = [1e-1 for i = 1:length(default_parameters)]
bounds = [(0.0, 2.0), (0.0, 1.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0)]

loss_function = TimeAveragedLossFunction(data, targets=11:20:length(data), fields=(:U, :V, :T))

nll = NegativeLogLikelihood(model, data, loss_function)

# Initialize sampler and NLL
covariance, chain = estimate_covariance(nll, default_parameters, standard_deviation, 
                                        BoundedNormalPerturbation, bounds, samples=1000, niterations=3)

@show optimal_link = optimal(chain)

visualize_realizations(model, data, loss_function.targets[[1, length(loss_function.targets)]], 
                       default_parameters, optimal_link.param)
