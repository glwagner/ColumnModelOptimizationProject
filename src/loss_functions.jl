nan2inf(err) = isnan(err) ? Inf : err

function initialize_forward_run(model, data, params, index)
    set!(model, params)
    set!(model, data, index)
    model.clock.iter = 0
    return nothing
end

get_weight(weights, j) = weights[j]
get_weight(::Nothing, j) = 1

"Returns a weighted sum of the absolute error over `fields` of `model` and `data`."
function weighted_error(fields::Tuple, weights, model, data, iᵈᵃᵗᵃ)
    total_err = zero(eltype(model.grid))

    for (kᶠⁱᵉˡᵈ, field) in enumerate(fields)
        field_err = absolute_error(getproperty(model.solution, field), getproperty(data, field)[iᵈᵃᵗᵃ])
        total_err += get_weight(weights, kᶠⁱᵉˡᵈ) * field_err # accumulate error
    end

    return total_err
end

"Returns the absolute error between `model.solution.field` and `data.field`."
weighted_error(field::Symbol, ::Nothing, model, data, iᵈᵃᵗᵃ) =
    absolute_error(getproperty(model.solution, field), getproperty(data, field)[iᵈᵃᵗᵃ])

"""
    struct TimeAveragedLossFunction{T, F, W}

A time-averaged loss function.
"""
struct TimeAveragedLossFunction{T, F, W}
    targets :: T
     fields :: F
    weights :: W
end

TimeAveragedLossFunction(; targets, fields, weights=nothing) =
    TimeAveragedLossFunction(targets, fields, weights)

function (loss::TimeAveragedLossFunction)(parameters, whole_model, data)

    # Initialize
    j¹ = loss.targets[1]
    initialize_forward_run(whole_model, data, parameters, j¹)
    time_averaged_error = zero(eltype(whole_model.grid))

    # Integrate error using trapezoidal rule
    ntargets = length(loss.targets)
    for i in 2:length(loss.targets)-1
        jⁱ = loss.targets[i]
        j⁻ = loss.targets[i-1]

        run_until!(whole_model.model, whole_model.Δt, data.t[jⁱ])

        interval = data.t[jⁱ] - data.t[j⁻]
        time_averaged_error += interval * weighted_error(loss.fields, loss.weights, whole_model,
                                                         data, i)
    end

    # Sum error from final target
    jᵉⁿᵈ = loss.targets[end]
    j⁻ = loss.targets[end-1]

    run_until!(whole_model.model, whole_model.Δt, data.t[jᵉⁿᵈ])

    interval = data.t[jᵉⁿᵈ] - data.t[j⁻]

    time_averaged_error += interval/2 * weighted_error(loss.fields, loss.weights, whole_model, 
                                                       data, length(loss.targets))

    # Divide by total length of time-interval
    time_averaged_error /= (data.t[jᵉⁿᵈ] - data.t[j¹])

    return nan2inf(time_averaged_error)
end
