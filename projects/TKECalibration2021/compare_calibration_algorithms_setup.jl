relative_weight_options = Dict(
                "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0)
)

function validation_loss_reduction(ce::CalibrationExperiment, parameters::FreeParameters)
    validation_loss = ce.validation.nll(parameters)
    calibration_loss = ce.calibration.nll(parameters)

    default_validation_loss = ce.validation.nll(ce.default_parameters)
    default_calibration_loss = ce.calibration.nll(ce.default_parameters)

    validation_loss_reduction = validation_loss/default_validation_loss
    println("Validation loss reduction: $(validation_loss_reduction)")
    println("Training loss reduction: $(calibration_loss/default_calibration_loss)")

    return validation_loss_reduction
end

function visualize_and_save(ce::CalibrationExperiment, parameters, directory)

        function get_Δt(Nt)
                Δt = 90
                if Nt > 400; Δt = 240; end
                if Nt > 800; Δt = 360; end
                return Δt
        end

        path = directory*"Plots/"
        mkpath(path)

        for LEScase in values(ce.calibration.LESdata) + values(ce.validation.LESdata)
                nll = get_nll(LESdata, RelevantParameters, ParametersToOptimize, relative_weights)
                set!(nll.model, parameters)
                Nt = length(nll.data)
                p = visualize_realizations(nll.model, nll.data, 60:get_Δt(Nt):length(nll.data), best_parameters, fields = nll.loss.fields)
                PyPlot.savefig(path*"$(Nt)_$(case_nll.data.name).png")
        end

end
