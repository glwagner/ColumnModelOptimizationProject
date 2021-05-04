using TKECalibration2021
using ColumnModelOptimizationProject
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using Distributions
using LinearAlgebra
using Random
using Dao
using Plots
using PyPlot
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--relative_weight_option"
        default = "all_but_e"
        arg_type = String
    "--free_parameters"
        default = "TKEParametersConvectiveAdjustmentRiDependent"
        arg_type = String
end
args = parse_args(s)
relative_weight_option = args["relative_weight_option"]
free_parameter_type = args["free_parameters"]

include("calibration_example_EKI_setup.jl")

p = Parameters(RelevantParameters = poptions[free_parameter_type], ParametersToOptimize = poptions[free_parameter_type])
calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options[relative_weight_option]);
validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options["all_but_e"]);
ce = CalibrationExperiment(calibration, validation, p)

directory = "EKI/$(free_parameter_type)_$(relative_weight_option)/"
isdir(directory) || mkpath(directory)

nll = ce.calibration.nll_wrapper
nll_validation = ce.validation.nll_wrapper
initial_parameters = ce.calibration.default_parameters
parameternames = propertynames(initial_parameters)

# plot_stds_within_bounds(nll, nll_validation, initial_parameters, directory, xrange=-3:0.25:5)
v = plot_prior_variance(nll, nll_validation, initial_parameters, directory; xrange=0.1:0.05:1.0)
# plot_num_ensemble_members(nll, nll_validation, initial_parameters, directory; xrange=1:5:30)
nl = plot_observation_noise_level(nll, nll_validation, initial_parameters, directory; xrange=-2.0:0.1:3.0)

# v, nl = plot_prior_variance_and_obs_noise_level(nll, nll_validation, initial_parameters, directory)
params, losses, mean_vars = eki(nll, initial_parameters; N_ens=15, N_iter=20, noise_level=nl, stds_within_bounds=v)
visualize_and_save(ce, ce.parameters.ParametersToOptimize(params), directory*"Visualize")

# Covariance
# params, losses, mean_vars = eki(nll, initial_parameters; N_ens=10, N_iter=20)
println(mean_vars)
println(ce.calibration.nll(initial_parameters))
println(ce.calibration.nll_wrapper(params))

# Parameter convergence
p = Plots.plot(title="Parameter Convergence", xlabel="Iteration", ylabel="Ensemble Covariance")
for pindex = 1:length(mean_vars[1])
    plot!(1:length(mean_vars), [x[pindex] for x in mean_vars], label=parameter_latex_guide[parameternames[pindex]], lw=4, legend=:topright)
end
Plots.savefig(p, directory*"covariance.pdf")

# Losses
p = Plots.plot(title="Loss on Ensemble Mean", xlabel="Iteration", ylabel="Loss")
plot!(1:length(losses), losses, lw=4, legend=:bottomright)
Plots.savefig(p, directory*"loss.pdf")
