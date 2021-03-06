using ColumnModelOptimizationProject

include("setup.jl")
include("utils.jl")

# Optimization parameters
        casename = "convection, N²: 2e-6"
         samples = 100
              Δz = 2.0
              Δt = 1minute
         LEScase = LESbrary[casename]

# Place to store results
results = @sprintf("kpp_calibration_%s_dz%d_dt%d.jld2", 
                   replace(replace(casename, ", " => "_"), ": " => ""),
                   Δz, Δt/minute) 

# Run the case
annealing = calibrate_kpp(joinpath(LESbrary_path, LEScase.filename), 
                                   samples = samples,
                                iterations = 3,
                                        Δz = Δz,
                                        Δt = Δt,
                              first_target = LEScase.first, 
                               last_target = LEScase.last,
                                    fields = kpp_fields(LEScase),
                          relative_weights = kpp_relative_weights(LEScase))
#                          profile_analysis = GradientProfileAnalysis(gradient_weight=0.5, value_weight=0.5))

# Save results
@save results annealing

# Do some simple analysis
model = annealing.negative_log_likelihood.model
 data = annealing.negative_log_likelihood.data
 loss = annealing.negative_log_likelihood.loss
chain = annealing.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:U, :T), 
                                          figsize = (16, 6)) 
