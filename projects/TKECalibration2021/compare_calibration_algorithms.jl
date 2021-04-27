## Optimizing TKE parameters
using TKECalibration2021
using Plots

@free_parameters(ConvectiveAdjustmentParameters,
                 Cᴬu, Cᴬc, Cᴬe)

relative_weights = Dict(
                "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0)
)

p = Parameters(RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent,
               ParametersToOptimize = ConvectiveAdjustmentParameters
              )

calibration = DataSet(FourDaySuite, p; relative_weights = relative_weights["all_e"])

typeof(relative_weights["all_e"])
using OrderedCollections
typeof(FourDaySuite) <: OrderedCollections.OrderedDict

validation = DataSet(merge(TwoDaySuite, SixDaySuite), p, relative_weights["uniform"])

ce = CalibrationExperiment(calibration, validation, p)

macro free_parameters(GroupName, parameter_names...)
    N = length(parameter_names)
    parameter_exprs = [:($name :: T; ) for name in parameter_names]
    return esc(quote
        Base.@kwdef mutable struct $GroupName{T} <: FreeParameters{$N, T}
            $(parameter_exprs...)
        end
    end)
end

initial_parameters = ParametersToOptimize([0.0057, 0.6706, 0.2717])
initial_parameters = ParametersToOptimize([0.0024, 0.7355, 1.4574])
initial_parameters = ParametersToOptimize([0.0057, 0.005, 0.005])

validation_loss_reduction(ce, initial_parameters)
ce.validation.nll(initial_parameters)

typeof(TKEParametersConvectiveAdjustmentRiIndependent) <: FreeParameters
parent(initial_parameters)

## Large search
# set_prior_means_to_initial_parameters = false
# stds_within_bounds = 5
# dname = "calibrate_FourDaySuite_validate_TwoDaySuiteSixDaySuite/prior_mean_center_bounds/$(relative_weights_option)_weights"

## Small search
set_prior_means_to_initial_parameters = true
stds_within_bounds = 3
dname = "calibrate_FourDaySuite_validate_TwoDaySuiteSixDaySuite/prior_mean_optimal/$(relative_weights_option)_weights"

# xs = collect(0.001:0.001:0.025)
# ll = Dict()
# for x in xs
#         initial_parameters.Cᴬc = 0.6706
#         ll[x] = nll_validation(initial_parameters)
# end
# Plots.plot(ll, yscale = :log10)
# argmin(ll)

##
# directory = pwd() * "/TKECalibration2021Results/compare_calibration_algorithms/$(dname)/$(RelevantParameters)/"
#
# o = open_output_file(directory)
#
# params_dict = Dict()
# loss_dict = Dict()
# function writeout2(o, name, params, loss, loss_validation)
#         param_vect = [params...]
#         write(o, "----------- \n")
#         write(o, "$(name) \n")
#         write(o, "Parameters: $(param_vect) \n")
#         write(o, "Loss on training: $(loss) \n")
#         write(o, "Loss on validation: $(loss_validation) \n")
#         params_dict[name] = param_vect
#         loss_dict[name] = loss_validation
# end
# writeout3(o, name, params) = writeout2(o, name, params, nll(params), nll_validation(params))
#
# @info "Output statistics will be written to: $(directory)"
#
# writeout3(o, "Default", initial_parameters)
#
# @info "Running Random Plugin..."
# random_plugin_params = random_plugin(nll, initial_parameters; function_calls=1000)
# writeout3(o, "Random_Plugin", random_plugin_params)
# println("Random Plugin", random_plugin_params)
#
# @info "Running Gradient Descent..."
# parameters = gradient_descent(nll, random_plugin_params; linebounds = (0, 100.0), linesearches = 10)
# writeout3(o, "Gradient_Descent", parameters)
# println("Gradient_Descent", parameters)
#

@info "Running Nelder-Mead from Optim.jl..."
parameters = nelder_mead(nll, initial_parameters)
writeout3(o, "Nelder_Mead", parameters)

nll(initial_parameters)
nll([parameters...])
nll_validation(initial_parameters)
nll_validation([parameters...])

@info "Running L-BFGS from Optim.jl..."
parameters = l_bfgs(nll, initial_parameters)
writeout3(o, "L_BFGS", parameters)

stds_within_bounds = 5
@info "Running Iterative Simulated Annealing..."
prob = simulated_annealing(nll, initial_parameters; samples = 500, iterations = 3,
                                initial_scale = 1e1,
                                final_scale = 1e-1,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = stds_within_bounds)
parameters = Dao.optimal(prob.markov_chains[end]).param
writeout3(o, "Annealing", parameters)
println([parameters...])



# initial_parameters = ParametersToOptimize([0.029469308779054255, 31.6606181722508, 416.89781702903394])
# propertynames(initial_parameters)
# nll(initial_parameters)
# nll_validation(initial_parameters)
# nll(ParametersToOptimize(parameters))
# nll_validation(ParametersToOptimize(parameters))
# println([parameters...])
#
# @info "Running Ensemble Kalman Inversion..."
parameters = ensemble_kalman_inversion(nll, initial_parameters; N_ens = 50, N_iter = 10,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = 10)
# writeout3(o, "EKI", ParametersToOptimize(parameters))

validation_losses = Dict()
initial_validation_loss = nll_validation(initial_parameters)
for x = 0.0:0.1:1.0
        println(x)
        relative_weights = Dict(:T => 1.0, :U => x, :V => x, :e => x)
        nll, _ = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize;
                                        loss_closure = loss_closure,
                                        relative_weights = relative_weights)
        prob = simulated_annealing(nll, initial_parameters; samples = 100, iterations = 3,
                                        initial_scale = 1e0,
                                        set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                        stds_within_bounds = stds_within_bounds)
        parameters = Dao.optimal(prob.markov_chains[end]).param
        println(parameters)
        loss_reduction =  nll_validation(ParametersToOptimize(parameters)) / initial_validation_loss
        validation_losses[x] = loss_reduction
        println(loss_reduction)
end
p = Plots.plot(validation_losses, ylabel="Validation loss reduction (Final / Initial)", xlabel="relative weight for U, V, e (where T -> 1)", title = "Loss reduction vs. U, V, e relative weight", legend=false, lw=3)
Plots.savefig(p, "____relative_weight_UVe.pdf")



prob = simulated_annealing(nll, initial_parameters; samples = 500, iterations = 3,
                                initial_scale = 1e1,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = stds_within_bounds)
parameters = Dao.optimal(prob.markov_chains[end]).param


propertynames(ParametersToOptimize(initial_parameters))

nll(initial_parameters)
nll_validation(initial_parameters)
println(parameters)
nll(parameters)
nll_validation(parameters)

initial_parameters = ParametersToOptimize([0.0057, 0.005, 0.005])

best_parameters = ParametersToOptimize(initial_parameters)
directory = "TKECalibration2021Results/annealing_visuals_smaller_CA/"


##
@info "Visualizing final parameter values for each calibration method..."
methodnames = [keys(params_dict)...]
parameters
# x-axis: sort calibration method names by the loss, highest to lowest (theoretically Default should come first)
sort_key(methodname) = loss_dict[methodname]
methodnames = sort([keys(loss_dict)...], by=sort_key, rev=true)

isdir(directory*"Parameters/") || mkdir(directory*"Parameters/")
for i in 1:length(parameters)
        paramname = propertynames(parameters)
        parameter_vals = [params_dict[name][i] for name in methodnames]
        p = Plots.plot(methodnames, parameter_vals, size=(600,150), linewidth=3, xrotation = 60, label=parameter_latex_guide[paramname[i]])
        Plots.savefig(directory*"Parameters/$(parameter_latex_guide[paramname[i]]).pdf")
end

@info "Visualizing final loss values for each calibration method..."

loss_vals = [loss_dict[name] for name in methodnames]
p = Plots.plot(methodnames, loss_vals, size=(600,150), linewidth=3, xrotation = 60, label="loss", yscale=:log10, color = :purple)
Plots.savefig(directory*"losses.png")

@info "Visualizing how the parameters perform on new data..."

# Find the overall best loss-minimizing parameter
best_method = argmin(loss_dict)
best_parameters = ParametersToOptimize(params_dict[best_method])

# best_method = "Nelder_Mead"
# best_parameters = ParametersToOptimize([0.494343889780388, 0.5671815040873687, 0.8034339015426114, 0.40412711476911073, 0.23935082563117294, 7.594543282811973, 0.19964793087118093, 3.01077631309058])

isdir(directory*"Test/") || mkdir(directory*"Test/")
isdir(directory*"Train/") || mkdir(directory*"Train/")
write(o, " \n")
write(o, "Best method: $(best_method)\n")
write(o, "Best loss: $(loss_dict[best_method])\n")
write(o, "Best parameters: \n")
write(o, "$(best_parameters) \n")

write(o, "Losses on Calibration Simulations: \n")
for LEScase in values(LESdata)
        case_nll, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_nll.data.name): $(case_nll(best_parameters)) \n")

        p = visualize_realizations(case_nll.model, case_nll.data, 12:180:length(case_nll.data), best_parameters)
        PyPlot.savefig(directory*"Train/$(case_nll.data.name).png")
end

write(o, "Loss on $(LESdata_validation) Validation Simulations: $(nll_validation(best_parameters))\n")
write(o, "Losses on Validation Simulations: \n")
for LEScase in values(LESdata_validation)
        case_nll, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_nll.data.name): $(case_nll(best_parameters)) \n")

        p = visualize_realizations(case_nll.model, case_nll.data, 12:180:length(case_nll.data), best_parameters)
        PyPlot.savefig(directory*"Test/$(case_nll.data.name).png")
end

# Close output.txt
close(o)

##
# initial_parameters = ParametersToOptimize([2.3923033609398985, 0.20312574763086733, 0.46858459323259577,
#                                            0.4460275753651033, 0.5207833999203864, 5.368922290345999,
#                                            1.1855706525110876, 2.6304133954266207])

# using above initial parameters, Nelder-Mead finds:
# initial_parameters = ParametersToOptimize([2.1638101987647502, 0.2172594537369187, 0.4522886369267623, 0.7534625713891345, 0.4477179760916435,
#         6.777679962252731, 1.2403584780163417, 1.9967245163343093])
