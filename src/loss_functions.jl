time_averaged_error(err, nerr) = isnan(err) ? Inf : err / nerr

function initialize_forward_run(column_model, column_data, params)
    set!(column_model, params)
    set!(column_model, column_data, column_data.initial)
    return zero(eltype(column_model.model.solution.U))
end

function temperature_loss(params, column_model, column_data)
    err = initialize_forward_run(column_model, column_data, params)
    # Run the model forward and collect error
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])
        err += absolute_error(column_model.model.solution.T, column_data.T[i])
    end
    return time_averaged_error(err, length(column_data.targets))
end

function velocity_loss(params, column_model, column_data)
    err = initialize_forward_run(column_model, column_data, params)
    # Run the model forward and collect error
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])
        err += absolute_error(column_model.model.solution.U, column_data.U[i])
        err += absolute_error(column_model.model.solution.V, column_data.V[i])
    end
    return time_averaged_error(err, length(column_data.targets))
end

function weighted_fields_loss(params, column_model, column_data, field_weights;
                                fields=(:U, :V, :T))
    total_err = initialize_forward_run(column_model, column_data, params)
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])

        for (j, field) in enumerate(fields)
            field_err = absolute_error(
                getproperty(column_model.model.solution, field),
                getproperty(column_data, field)[i])
            total_err += field_weights[j] * field_err # accumulate error
        end
    end
    return time_averaged_error(total_err, length(column_data.targets))
end

function relative_fields_loss(params, column_model, column_data;
                                fields=(:U, :V, :T))
    total_err = initialize_forward_run(column_model, column_data, params)
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])

        for (j, field) in enumerate(fields)
            field_err = relative_error(
                getproperty(column_model.model.solution, field),
                getproperty(column_data, field)[i])
            total_err += field_err # accumulate error
        end
    end
    return time_averaged_error(total_err, length(column_data.targets))
end
