using 
    OceanTurb,
    Dao,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization

datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

data = ColumnData(datapath)

model = ModularKPPOptimization.ColumnModel(data, 5minute, N=64, 
                                           mixingdepth=ModularKPP.ROMSMixingDepth(),
                                           kprofile=ModularKPP.GeneralizedCubicPolynomial())

default_parameters = DefaultFreeParameters(model, WindMixingAndShapeParameters)
standard_deviation = Array(1e-1 * default_parameters)
bounds = [(0.0, 2.0), (0.0, 1.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0)]

#loss_function = TimeAveragedLossFunction(targets=11:10:201, fields=(:U, :V, :T), weights=(0.0, 0.0, 0.1))
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

#=
for i = 1:iterations

    global chain
    global covariance

    if i == 1
        initial_link = MarkovLink(nll, default_parameters)
        sampler = MetropolisSampler(BoundedNormalPerturbation(standard_deviation, bounds))
        @show nll.scale = initial_link.error
        initial_link = MarkovLink(nll, default_parameters)
    else
        optimal_link = optimal(chain)
        sampler = MetropolisSampler(BoundedNormalPerturbation(covariance, bounds))
        @show nll.scale = optimal_link.error
        initial_link = MarkovLink(nll, optimal_link.param)
    end

    @time chain = MarkovChain(i * initial_steps, initial_link, nll, sampler)
    parameter_matrix = zeros(3, length(chain))

    for (i, link) in enumerate(chain.links)
        parameter_matrix[:, i] .= link.param
    end

    @show covariance = cov(parameter_matrix, dims=2)
    @show optimal(chain).error / nll.scale

end
=#

#=
@show chain.acceptance
@show optimal(chain)

visualize_realizations(model, data, loss_function.targets[[1, 401]], chain[1].param, optimal(chain).param)
=#
