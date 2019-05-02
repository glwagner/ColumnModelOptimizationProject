function temperature_loss(params, column_model, column_data)

    # Prepare the model by setting parameters and initial conditions
    set!(column_model, params)
    set!(column_model, column_data, column_data.initial)

    err = zero(eltype(column_model.model.solution.U))

    # Run the model forward and collect error
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])
        err += absolute_error(column_model.model.solution.T, column_data.T[i])
    end

    if isnan(err)
        err = Inf
    end

    return err / length(column_data.targets)
end

function weighted_fields_loss(params, column_model, column_data,
                              field_weights; fields=(:U, :V, :T, :S))

    # Prepare the model by setting parameters and initial conditions
    set!(column_model, params)
    set!(column_model, column_data, column_data.initial)

    total_err = zero(eltype(column_model.model.solution.U))

    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])

        for (j, field) in enumerate(fields)
            field_err = absolute_error(
                getproperty(column_model.model.solution, field),
                getproperty(column_data, field)[i])

            # Accumulate error
            total_err += field_weights[j] * field_err
        end
    end

    if isnan(total_err)
        total_err = Inf
    end

    return total_err / length(column_data.targets)
end
