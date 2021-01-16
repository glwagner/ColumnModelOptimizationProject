using 
    OceanTurb,
    PyPlot,
    Dao,
    Distributions,
    ColumnModelOptimizationProject

using Statistics: mean

using ColumnModelOptimizationProject.TKEMassFluxOptimization

datadir = "/Users/andresouza/Dropbox/greg-andre/BoundaryLayerTurbulenceData/"

#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
#dataname = "stress_driven_Nsq1.6e-05_f0.0e+00_Qu1.0e-04_Nh128_Nz128_averages.jld2"
dataname = "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2"

datapath = joinpath(datadir, dataname)

# Model and data
data = ColumnData(datapath)
model = TKEMassFluxOptimization.ColumnModel(data, 1minute, N=64,
                                            mixing_length=TKEMassFlux.SimpleMixingLength())
                                            #mixing_length=TKEMassFlux.EquilibriumMixingLength())

# Create loss function and negative-log-likelihood object
targets = 21:11:length(data)
fields = (:T, :U, :e)

max_variances = [max_variance(data, field) for field in fields]
weights = [1/σ for σ in max_variances]
weights[1] *= 100
weights[2] *= 10

loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights)
nll = NegativeLogLikelihood(model, data, loss)

@free_parameters ParametersToOptimize Cᴷu Cᴷe CᴷPr Cᴰ Cᴸʷ Cʷu★

# Initial state for optimization step
default_parameters = DefaultFreeParameters(model, ParametersToOptimize)

variance = ParametersToOptimize((1e-1 for p in default_parameters)...)
variance = Array(variance)

bounds = ParametersToOptimize(((0.0, 10.0) for p in default_parameters)...)

# Iterative simulated annealing...
nsamples = 1000
niters = 20

covariance, chains = anneal(nll, default_parameters, variance, BoundedNormalPerturbation, bounds;
             iterations = niters,
                samples = iter -> round(Int, sqrt(iter) * nsamples),
     annealing_schedule = (nll, iter, link) -> 10 * niters/iter * link.error,
    covariance_schedule = (Σ, iter) -> 1 .* Σ ./ iter
)

# Re-estimate parameter bounds
bounds = estimate_bounds(chains[end])
new_parameters = optimal(chains[end]).param

#=
# Try again.
nsamples = 10000
niters = 8

covariance, chains = anneal(nll, new_parameters, covariance, BoundedNormalPerturbation, bounds;
                                        samples = nsamples,
                                     iterations = niters,
                             annealing_schedule = (nll, iter, link) -> link.error / iter,
                            covariance_schedule = (Σ, iter) -> Σ ./ iter)
=#


@show optimal_link = optimal(chains[end])

optimal_parameters = optimal_link.param

viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], default_parameters,
                                  optimal_parameters, fields=(:U, :T, :e), figsize=(24, 36))

chain = chains[end]
parameters_to_plot = propertynames(default_parameters) #(:Cᴰ, :Cᴷᵤ, :Cᴷₑ, :Cᴾʳ, :Cʷu★)
pdf_fig, pdf_axs = subplots(nrows=length(parameters_to_plot), figsize=(10, 36))

for i = 1:3

    global viz_fig

    extend!(chain, nsamples)

    @show chain optimal(chain)

    viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], default_parameters, 
                                              optimal(chain).param, fields=(:U, :T, :e), fig=viz_fig)
                                          
    for ax in pdf_axs
        sca(ax); cla()
    end

    for (i, parameter) in enumerate(parameters_to_plot)
        visualize_markov_chain!(pdf_axs[i], chain, parameter)
        sca(pdf_axs[i])
        xlabel(parameter_latex_guide[parameter])
    end

    tight_layout()
    pause(0.1)

end
