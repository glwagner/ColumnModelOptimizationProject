using TKECalibration2021
using StatsPlots
using Distributions
using LinearAlgebra
using Random
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using ArgParse
using ColumnModelOptimizationProject
using Plots
using Dao
using PyPlot

s = ArgParseSettings()
@add_arg_table! s begin
    "--relative_weight_option"
        help = ""
        default = "all_but_e"
        arg_type = String
end
relative_weight_option = parse_args(s)["relative_weight_option"]

directory = "Distributions/TKEParametersConvectiveAdjustmentRiDependent/"
isdir(directory) || mkdir(directory)

# @free_parameters(ConvectiveAdjustmentParameters,
#                  Cᴬu, Cᴬc, Cᴬe)

relative_weight_options = Dict(
                "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0),
                "all_but_e" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 0.0),
                "all_uv" => Dict(:T => 0.0, :U => 1.0, :V => 1.0, :e => 0.0)
)

p = Parameters(RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent,
               ParametersToOptimize = TKEParametersConvectiveAdjustmentRiIndependent
              )

# relative_weight_option = "uniform"
calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options[relative_weight_option]);
validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options[relative_weight_option]);
ce = CalibrationExperiment(calibration, validation, p)

nll = ce.calibration.nll_wrapper
nll_validation = ce.validation.nll_wrapper
initial_parameters = ce.calibration.default_parameters

# nll_validation([initial_parameters...])

ce.parameters.RelevantParameters([initial_parameters...])
propertynames(initial_parameters)

function get_losses(pvalues, pname, loss)
    defaults = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
    ℒvalues = []
    for pvalue in pvalues
        TKECalibration2021.set_if_present!(defaults, pname, pvalue)
        # println(defaults)
        push!(ℒvalues, loss([defaults...]))
    end
    return ℒvalues
end


# pname = :Cᴬu
# pvalues = range(0.0, stop=3.0, length=1000)
# pvalues = range(0.001, stop=0.1, length=99)
# loss = nll
# a = get_losses(pvalues, pname, loss)
# pvalues[argmin(a)]
# Plots.plot(pvalues, a)

function lognormal_μ_σ²(means, variances)
    μs = []
    σ²s = []
    for i in 1:length(means)
        k = variances[i]/(means[i]^2) + 1
        # μ = log(means[i]/sqrt(k))
        # μ = log(means[i]/sqrt(k))
        μ = log(means[i]/sqrt(k))
        σ² = log(k)
        push!(μs, μ)
        push!(σ²s, σ²)
    end
    return μs, σ²s
end

println(initial_parameters)
bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = 5);
prior_means = [initial_parameters...]
μs, σ²s = lognormal_μ_σ²(prior_means, prior_variances)

# first term (data misfit) of EKI objective = ℒ / obs_noise_level
ℒ = ce.calibration.nll_wrapper(prior_means)
println(ℒ)

# second term (prior misfit) of EKI objective = || σ²s.^(-0.5) .* μs ||²
pr = norm((σ²s.^(-1/2)) .* μs)^2
println(pr)

# for equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about
obs_noise_level = ℒ / pr

# obs = 1e-3
# ℒ / (obs*pr) = 100
# obs = ℒ / (100 * pr)

all_plots = []
for i in 1:length(initial_parameters)
    pname = propertynames(initial_parameters)[i]
    pvalue_lims = (max(0.0, prior_means[i]-sqrt(prior_variances[i])), prior_means[i]+2*sqrt(prior_variances[i]))
    pvalues = range(pvalue_lims[1],stop=pvalue_lims[2],length=5)
    losses_cal = get_losses(pvalues, pname, nll)
    losses_val = get_losses(pvalues, pname, nll_validation)

    kwargs = (lw=4, xlims = pvalue_lims, xlabel = "$(parameter_latex_guide[pname])")
    distplot = StatsPlots.plot(LogNormal(μs[i], σ²s[i]); ylabel = L"P_{prior}(\theta)", label="", color=:red, kwargs...)
    plot!([prior_means[i]], linetype = :vline, linestyle=:dash, color=:red, label="mean")

    lossplot = Plots.plot(pvalues, log.(losses_val), label="val", color=:blue, lw=4, la=0.5)
    plot!(pvalues, log.(losses_cal);  xlabel="$(parameter_latex_guide[pname])", color=:green, la=0.5, label="cal", ylabel = L"\log_{10}\mathcal{L}(\theta)", kwargs...)
    plot!([pvalues[argmin(losses_cal)]], linetype = :vline, linestyle=:dash, color=:blue, la=0.5, label="min")
    plot!([pvalues[argmin(losses_val)]], linetype = :vline, linestyle=:dash, color=:green, la=0.5, label="min")

    layout = @layout [a;b]
    pplot = Plots.plot(distplot, lossplot; layout=layout, framestyle=:box)
    Plots.savefig(directory*"$(pname)_$(relative_weight_option).png")
    push!(all_plots, pplot)
end
Plots.plot(all_plots..., layout=(3,5), size=(1600,1200), left_margin=10*Plots.mm, bottom_margin=10*Plots.mm)
Plots.savefig(directory*"parameters_$(relative_weight_option).pdf")
