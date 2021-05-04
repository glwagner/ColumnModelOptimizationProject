## Optimizing TKE parameters
using TKECalibration2021
using ColumnModelOptimizationProject
using Plots
using Dao
using PyPlot

# @free_parameters(ConvectiveAdjustmentParameters,
#                  Cᴬu, Cᴬc, Cᴬe)

relative_weight_options = Dict(
                "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0),
                "all_but_e" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 0.0),
                "all_uv" => Dict(:T => 0.0, :U => 1.0, :V => 1.0, :e => 0.0)
)

p = Parameters(RelevantParameters = TKEParametersConvectiveAdjustmentRiDependent,
               ParametersToOptimize = TKEParametersConvectiveAdjustmentRiDependent
              )

calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"]);
validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options["uniform"]);

ce = CalibrationExperiment(calibration, validation, p)

function validation_loss_reduction(ce::CalibrationExperiment, parameters::FreeParameters)
    validation_loss = ce.validation.nll(parameters)
    calibration_loss = ce.calibration.nll(parameters)

    default_validation_loss = ce.validation.nll(ce.default_parameters)
    default_calibration_loss = ce.calibration.nll(ce.default_parameters)

    validation_loss_reduction = validation_loss/default_validation_loss
    println("Parameters: $([parameters...])")
    println("Validation loss reduction: $(validation_loss_reduction)")
    println("Training loss reduction: $(calibration_loss/default_calibration_loss)")

    return validation_loss_reduction
end
