using Statistics, Distributions, PyPlot
using OceanTurb, Dao, ColumnModelOptimizationProject

using ColumnModelOptimizationProject.TKEMassFluxOptimization
using ColumnModelOptimizationProject.TKEMassFluxOptimization: ColumnModel

#dataname = "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2"; first_target=21; last_target=nothing
#dataname = "kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx256_Nz256_averages.jld2"; first_target=21; last_target=nothing
#dataname = "kato_phillips_Nsq1.0e-06_Qu1.0e-04_Nx512_Nz256_averages.jld2"; first_target=5; last_target=61
# dataname = "kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx256_Nz256_averages.jld2"; first_target=5; last_target=51
# datadir = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
# datapath = joinpath(datadir, dataname)

datapath = "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2"

# Model and data
column_data = ColumnData(datapath)
model = ColumnModel(column_data, 1minute, N=32,
                    mixing_length = TKEMassFlux.SimpleMixingLength(),
                    tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux())
                    #tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEValue())
                    #mixing_length=TKEMassFlux.EquilibriumMixingLength())

first_target = 1
last_target = length(column_data)
# Create loss function and negative-log-likelihood object
last_target = last_target === nothing ? length(column_data) : last_target
@show targets = first_target:last_target
# fields = (:T, :U, :V, :e)
fields = (:T,)

# Estimate weights based on maximum variance in the data
max_variances = [max_variance(column_data, field, targets) for field in fields]
weights = [1/σ for σ in max_variances]
weights[1] *= 1
weights[1] = 1

@show fields weights

loss = LossFunction(model, column_data, fields=fields, targets=targets, weights=weights)
nll = NegativeLogLikelihood(model, column_data, loss)

# @free_parameters ParametersToOptimize Cᴸʷ Cᴸᵇ Cᴰ Cᴷᵤ Cᴾʳ Cᴷₑ Cʷu★
@free_parameters ParametersToOptimize Cᴷu Cᴷe Cᴰ Cʷu★ Cᴸᵇ

# Initial state for optimization step
default_parameters = DefaultFreeParameters(model, ParametersToOptimize)

# Set bounds on free parameters
bounds = ParametersToOptimize(((0.01, 2.0) for p in default_parameters)...)
bounds.Cᴷu  = (0.01, 0.5)
bounds.Cᴷe  = (0.01, 1.0)
bounds.Cᴰ   = (0.01, 1.0)
bounds.Cʷu★ = (0.01, 10.0)

variance = ParametersToOptimize((0.1 * bound[2] for bound in bounds)...)
variance = Array(variance)

# Iterative simulated annealing...
prob = anneal(nll, default_parameters, variance, BoundedNormalPerturbation, bounds;
             iterations = 10,
                samples = 1000,
     annealing_schedule = AdaptiveExponentialSchedule(initial_scale=100.0, final_scale=1e-3, convergence_rate=1.0),
    covariance_schedule = AdaptiveExponentialSchedule(initial_scale=1.0,   final_scale=1e-3, convergence_rate=0.1),
)
optimal(prob.markov_chains[end]).param

viz_fig, viz_axs = visualize_realizations(model, column_data, loss.targets[[1, end]],
                                          optimal(prob.markov_chains[1]).param,
                                          optimal(prob.markov_chains[end]).param,
                                          fields=(:U, :T, :e),
                                          figsize=(24, 36))

chain = prob.markov_chains[end]
parameters_to_plot = propertynames(default_parameters)

pdf_fig, pdf_axs = subplots(nrows=length(parameters_to_plot), figsize=(10, 36))

savefig("pdf_fig.pdf")
matplotlib.use("MacOSX")

for ax in pdf_axs
    sca(ax); cla()
end

for (i, parameter) in enumerate(parameters_to_plot)
    visualize_markov_chain!(pdf_axs[i], chain, parameter, alpha=0.2, facecolor="b")

    sca(pdf_axs[i])
    xlabel(parameter_latex_guide[parameter])
end

pause(0.1)
