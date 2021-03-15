## Optimizing TKE parameters

using Statistics, Distributions, PyPlot
using OceanTurb, ColumnModelOptimizationProject, Dao
using ColumnModelOptimizationProject.TKEMassFluxOptimization
using ColumnModelOptimizationProject.TKEMassFluxOptimization: ColumnModel

datapath = "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2"

# Model and data
column_data = ColumnData(datapath)
model = ColumnModel(column_data, 1minute, N=32,
                    mixing_length = TKEMassFlux.SimpleMixingLength(),
                    tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux())
                    #tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEValue())
                    #mixing_length=TKEMassFlux.EquilibriumMixingLength())

# Parameters
get_free_parameters(model)
@free_parameters ParametersToOptimize Cᴷu Cᴷe Cᴰ Cʷu★ Cᴸᵇ
default_parameters = DefaultFreeParameters(model, ParametersToOptimize)

# Set bounds on free parameters
bounds = ParametersToOptimize(((0.01, 2.0) for p in default_parameters)...)
bounds.Cᴷu  = (0.01, 0.5)
bounds.Cᴷe  = (0.01, 1.0)
bounds.Cᴰ   = (0.01, 1.0)
bounds.Cʷu★ = (0.01, 10.0)

# Create loss function and negative-log-likelihood object
loss_function = LossFunction(model, column_data,
                            fields=(:T,),
                            targets=1:length(column_data),
                            weights=[1.0,],
                            time_series = TimeSeriesAnalysis(column_data.t[targets], TimeAverage()),
                            profile = ValueProfileAnalysis(model.grid)
                            )

# This version of the loss runs the forward model internally
loss(default_parameters)

# Run forward map and then loss
# ℱ = forward_map(default_parameters, model, column_data, loss_function)
# myloss(ℱ) = loss_function(ℱ, column_data)
# myloss(ℱ)

@info "Optimizing TKE parameters..."

include("line_search_gradient_descent.jl")

# First construct global search
# Create Prior
priors = [Uniform(b...) for b in bounds]
# Determine number of function calls
functioncalls = 1000
# Define Method
method = RandomPlugin(priors, functioncalls)
# Optimize
minparam = optimize(loss, method)

# Next do gradient descent
# construct numerical gradient
∇loss(params) = gradient(loss, params)
# optimize choosing minimum from the global search for refinement
best_params = minparam
# method  = RandomLineSearch(linebounds = (0, 1e-0/norm(∇loss(best_params))), linesearches = 20)
method  = RandomLineSearch(linebounds = (0, 1.0), linesearches = 20)
bestparam = optimize(loss, ∇loss, best_params, method)

params = best_params
params = default_parameters
plot = visualize_realizations(model, column_data, 1:50:200, params)
plot
PyPlot.savefig("default_params.png")
