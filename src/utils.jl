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

nan2inf(err) = isnan(err) ? Inf : err

function trapz(f, t)
    @inbounds begin
        integral = zero(eltype(t))
        for i = 2:length(t)
            integral += (f[i] + f[i-1]) * (t[i] - t[i-1])
        end
    end
    return integral
end

function initialize_forward_run!(model, data, params, index)
    set!(model, params)
    set!(model, data, index)
    model.clock.iter = 0
    return nothing
end

struct VarianceWeights{F, D, T, V}
       fields :: F
         data :: D
      targets :: T
    variances :: V
end

@inbounds normalize_variance(::Nothing, field, σ) = σ

function VarianceWeights(data; fields, targets=1:length(data), normalizer=nothing)
    variances = (; zip(fields, (zeros(length(targets)) for field in fields))...)

    for (k, field) in enumerate(fields)
        for i in 1:length(targets)
            @inbounds variances[k][i] = normalize_variance(normalizer, field, variance(data, field, i))
        end
    end

    return VarianceWeights(fields, data, targets, variances)
end

