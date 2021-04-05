using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

casename = "free_convection"
# relative_weights = [1e+0, 1e-4, 1e-4, 1e-4]
LEScase = FourDaySuite[casename]
datapath = joinpath(FourDaySuite_path, LEScase.filename)

RelevantParameters = TKEConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEConvectiveAdjustmentRiIndependent

# Place to store results
# results = @sprintf("tke_calibration_%s_dz%d_dt%d.jld2",
#                    replace(replace(casename, ", " => "_"), ": " => ""),
#                    Δz, Δt/minute)

nll = init_tke_calibration(datapath;
                                         N = 32,
                                        Δt = 60.0, #1minute
                              first_target = LEScase.first,
                               last_target = LEScase.last,
                                    fields = tke_fields(LEScase),
                          relative_weights = tke_relative_weights(LEScase),
                        eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                     convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment(),
                        )

model = nll.model
cdata = nll.data
set!(model, custom_defaults(model, RelevantParameters))
initial_parameters = custom_defaults(model, ParametersToOptimize)

# Run the case
calibration = calibrate(nll, initial_parameters, samples = 1000, iterations = 10)

# Save results
# @save results calibration

# Do some simple analysis
 loss = calibration.negative_log_likelihood.loss
chain = calibration.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:T, :e),
                                          figsize = (16, 6))
