using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization: ColumnModel, WindMixingParameters

struct TestParameters{T} <: FreeParameters{1, T}
    CDe :: T
end

Base.similar(p::TestParameters{T}) where T = TestParameters{T}(0)

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
data = ColumnData(datapath)
model = ColumnModel(data, 5minute, N=64)

@show default_parameters = DefaultFreeParameters(model, WindMixingParameters)
standard_deviation = [1e-1 for i = 1:length(default_parameters)]
bounds = [(0.0, 3.0) for i = 1:length(default_parameters)]

loss_function = TimeAveragedLossFunction(data, targets=21:20:length(data), fields=(:U, :V, :T))

nll = NegativeLogLikelihood(model, data, loss_function)

@show variances = max_variance(data, loss_function)

#=
# Estimate the covariance matrix
covariance, chain = estimate_covariance(nll, default_parameters, standard_deviation, 
                                        BoundedNormalPerturbation, bounds, 
                                        samples=iter->1000, niterations=2)

@show optimal_link = optimal(chain)

visualize_realizations(model, data, loss_function.targets[[1, length(loss_function.targets)]], 
                       default_parameters, optimal_link.param)
                       =#
