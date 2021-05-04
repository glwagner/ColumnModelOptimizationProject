"""
Command line:
include("quick_setup.jl"); include("quick_calibrate.jl")
"""
loss = calibration.nll_wrapper
initial_parameters = ce.default_parameters

initial_parameters = ce.parameters.ParametersToOptimize([0.18001246435585952, 0.004894552755448252, 0.16896707401665273, 0.3477677143004802, 1.087806049439474, 9.426487502876356, 0.10722309745416062, 0.008382544820088256, 4.193596262599441, 1.9974808352399316])


println("Initial validation loss: $(ce.validation.nll(initial_parameters))")
validation_loss_reduction(ce, initial_parameters)

## Small search
set_prior_means_to_initial_parameters = true
stds_within_bounds = 5

# Leads to negative values
# @info "Running Nelder-Mead from Optim.jl..."
# parameters = nelder_mead(loss, initial_parameters)
# println(parameters)
# validation_loss_reduction(ce, calibration.parameters.ParametersToOptimize(parameters))

# @info "Running Iterative Simulated Annealing..."
# prob = simulated_annealing(calibration.nll, initial_parameters; samples = 1000, iterations = 5,
#                                 initial_scale = 1e0,
#                                 final_scale = 1e-2,
#                                 set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
#                                 stds_within_bounds = stds_within_bounds)
# parameters = Dao.optimal(prob.markov_chains[end]).param
# validation_loss_reduction(ce, parameters)

using Dates
directory = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
visualize_and_save(ce, parameters, directory)
