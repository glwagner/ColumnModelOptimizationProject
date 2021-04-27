function variance(data, field_name, i)
    field = getproperty(data, field_name)[i]
    field_mean = mean(field.data)

    variance = zero(eltype(field))
    for j in eachindex(field)
        @inbounds variance += (field[j] - field_mean)^2 * Δf(field, j)
    end

    return variance / height(field)
end

function gradient_variance(data, field_name, i)
    field = getproperty(data, field_name)[i]
    ∇field = ∂z(field)
    ∇field_mean = mean(∇field.data)

    variance = zero(eltype(field))
    for j = 2:field.grid.N # don't include end points
        @inbounds variance += (∇field[j] - ∇field_mean)^2 * Δc(field, j)
    end

    return variance / (height(field) - Δc(field, field.grid.N+1))
end

function max_variance(data, field_name, targets=1:length(data.t))
    maximum_variance = 0.0

    for target in targets
        maximum_variance = max(maximum_variance, variance(data, field_name, target))
    end

    return maximum_variance
end

function mean_variance(data, field_name, targets=1:length(data.t))
    total_variance = 0.0

    for target in targets
        total_variance += variance(data, field_name, target)
    end

    return total_variance / length(targets)
end

function max_gradient_variance(data, field_name, targets=1:length(data.t))
    maximum_variance = 0.0

    for target in targets
        maximum_variance = max(maximum_variance, gradient_variance(data, field_name, target))
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

function simple_safe_save(savename, variable, name="calibration")

    temppath = savename[1:end-5] * "_temp.jld2"
    newpath = savename

    isfile(newpath) && mv(newpath, temppath, force=true)

    println("Saving to $savename...")
    save(newpath, name, variable)

    isfile(temppath) && rm(temppath)

    return nothing
end
