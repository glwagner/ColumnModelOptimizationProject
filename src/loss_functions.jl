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

@inline get_weight(weights, k) = weights[k]
@inline get_weight(::Nothing, k) = 1

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
    struct LossFunction{A, T, F, W, S, M}

A loss function that computes an `analysis` of type `A` on an error time series
generated from multiple fields.
"""
struct LossFunction{A, T, F, W, S, M}
   analysis :: A
    targets :: T
     fields :: F
    weights :: W
      error :: S
       time :: M
end

LossFunction(analysis, data; fields, targets=1:length(data.t), weights=nothing) =
    LossFunction(analysis, targets, fields, weights, zeros(length(targets)), [data.t[i] for i in targets])

function evaluate_error_time_series!(loss, parameters, whole_model, data)

    # Initialize
    initialize_forward_run!(whole_model, data, parameters, loss.targets[1])
    loss.error[1] = 0.0

    # Calculate a time-seris of the error
    for i in 2:length(loss.targets)
        j = loss.targets[i]
        run_until!(whole_model.model, whole_model.Δt, data.t[j])
        @inbounds loss.error[i] = weighted_error(loss.fields, loss.weights, whole_model, data, i)
    end

    return nothing
end

const evaluate! = evaluate_error_time_series!

#
# Analysis types
#

struct TimeAverage end

const TimeAveragedLossFunction = LossFunction{<:TimeAverage}
TimeAveragedLossFunction(args...; kwargs...) = LossFunction(TimeAverage(), args...; kwargs...)

function (loss::TimeAveragedLossFunction)(parameters, whole_model, data)
    evaluate_error_time_series!(loss, parameters, whole_model, data)
    return trapz(loss.error, loss.time) / (loss.time[end] - loss.time[1])
end
