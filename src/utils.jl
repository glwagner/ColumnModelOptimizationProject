function variance(data, field_name, i)
    field = getproperty(data, field_name)[i]
    field_mean = mean(field.data)

    variance = zero(eltype(field))
    for j in eachindex(field)
        @inbounds variance += (field[j] - field_mean)^2 * Δf(field, j)
    end

    return variance
end

function max_variance(data, field_name, targets=1:length(data.t))
    maximum_variance = 0.0

    for target in targets
        field = getproperty(data, field_name)[target]
        fieldmean = mean(field.data)
        maximum_variance = max(maximum_variance, variance(data, field_name, target))
    end

    return maximum_variance
end

function initialize_and_run_until!(model, data, parameters, initial, target)
    initialize_forward_run!(model, data, parameters, initial)
    run_until!(model.model, model.Δt, data.t[target])
    return nothing
end
