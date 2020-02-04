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


@inline get_weight(weights, field_index, target_index) = @inbounds weights[field_index]
@inline get_weight(::Nothing, field_index, target_index) = 1
@inline get_weight(weights::VarianceWeights, field_index, target_index) = 
    @inbounds weights.variances[field_index][target_index]

@inline squared_absolute_error(model_field, data_field) = absolute_error(model_field, data_field)^2

"Returns a weighted sum of the absolute error over `fields` of `model` and `data`."
function weighted_error(fields::Tuple, weights, model, data, target_index)
    total_err = zero(eltype(model.grid))

    for (field_index, field) in enumerate(fields)
        field_err = squared_absolute_error(getproperty(model.solution, field), 
                                           getproperty(data, field)[target_index])

        total_err += get_weight(weights, field_index, target_index) * field_err # accumulate error
    end

    return nan2inf(total_err)
end

"Returns the absolute error between `model.solution.field` and `data.field`."
weighted_error(field::Symbol, ::Nothing, model, data, target_index) =
    nan2inf(squared_absolute_error(getproperty(model.solution, field), getproperty(data, field)[target_index]))

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

function max_variance(data, loss::LossFunction)
    max_variances = zeros(length(loss.fields))
    for (ifield, field) in enumerate(loss.fields)
        max_variances[ifield] = get_weight(weight, ifield) * max_variance(data, field, loss.targets)
    end
    return max_variances
end

function evaluate_error_time_series!(loss, parameters, whole_model, data)

    # Initialize
    initialize_forward_run!(whole_model, data, parameters, loss.targets[1])
    @inbounds loss.error[1] = weighted_error(loss.fields, loss.weights, whole_model, data, 1)

    # Calculate a time-series of the error
    for target_index in 2:length(loss.targets)
        target = loss.targets[target_index]
        run_until!(whole_model.model, whole_model.Δt, data.t[target])

        @inbounds loss.error[target_index] = 
            weighted_error(loss.fields, loss.weights, whole_model, data, target_index)
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
