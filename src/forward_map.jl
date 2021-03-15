# Evaluate loss function AFTER the forward map rather than during
mutable struct ForwardMapOutput{UU, VV, TΘ, SS, EE}
                     U :: UU
                     V :: VV
                     T :: TΘ
                     S :: SS
                     e :: EE
end

"""
Returns the evolutions of U, V, T, and e profiles.
"""
function forward_map(parameters, model_plus_Δt, data, loss) # loss is needed for targets -- that's all

    initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])

    forward_map_output = Dict(fieldname => Dict() for fieldname in loss.fields)
    for (i, target) in enumerate(loss.targets)
        run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[target])
        for fieldname in loss.fields
            forward_map_output[fieldname][target] = deepcopy(getproperty(model_plus_Δt.solution, fieldname))
        end
    end

    return forward_map_output
end


# """
# Returns the evolutions of U, V, T, and e profiles.
# """
# function forward_map(parameters, model_plus_Δt, data, loss) # loss is needed for targets -- that's all
#     empty = [model_plus_Δt.solution.T for x in loss.targets]
#     empty = OffsetVector(empty, loss.targets)
#     forward_map_output = ForwardMapOutput(empty, empty, empty, empty, empty)
#     initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])
#
#     for (i, target) in enumerate(loss.targets)
#         run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[target])
#         @. forward_map_output.U[target].data = getproperty(model_plus_Δt.solution, :U).data
#         @. forward_map_output.V[target].data = getproperty(model_plus_Δt.solution, :V).data
#         @. forward_map_output.T[target].data = getproperty(model_plus_Δt.solution, :T).data
#         @. forward_map_output.S[target].data = getproperty(model_plus_Δt.solution, :S).data
#         @. forward_map_output.e[target].data = getproperty(model_plus_Δt.solution, :e).data
#     end
#     # for (field_index, field_name) in enumerate(field_names)
#     #     @. forward_map_output = getproperty(model_plus_Δt.solution, field_name).data[target]
#     #     data_field = getproperty(data, field_name)[target]
#
#     println("$(mean(forward_map_output.e[1])) | $(mean(forward_map_output.e[33])) | $(mean(forward_map_output.e[289]))")
#     println("$(mean(forward_map_output.T[1])) | $(mean(forward_map_output.T[33])) | $(mean(forward_map_output.T[289]))")
#     return forward_map_output
#     # initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])
#     #
#     # U = deepcopy(empty)
#     # V = deepcopy(empty)
#     # T = deepcopy(empty)
#     # S = deepcopy(empty)
#     # e = deepcopy(empty)
#     #
#     # for (i, target) in enumerate(loss.targets)
#     #     run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[target])
#     #     @. U[i].data = model_plus_Δt.solution.U.data
#     #     @. V[i].data = model_plus_Δt.solution.V.data
#     #     @. T[i].data = model_plus_Δt.solution.T.data
#     #     @. S[i].data = model_plus_Δt.solution.S.data
#     #     @. e[i].data = model_plus_Δt.solution.e.data
#     # end
#     #
#     # println("$(mean(forward_map_output.e[1])) | $(mean(forward_map_output.e[33])) | $(mean(forward_map_output.e[289]))")
#     # println("$(mean(forward_map_output.T[1])) | $(mean(forward_map_output.T[33])) | $(mean(forward_map_output.T[289]))")
#     # U = OffsetVector(U, loss.targets)
#     # V = OffsetVector(V, loss.targets)
#     # T = OffsetVector(T, loss.targets)
#     # S = OffsetVector(S, loss.targets)
#     # e = OffsetVector(e, loss.targets)
#     #
#     # return ForwardMapOutput(U, V, T, S, e)
# end

# Evaluate loss function AFTER the forward map rather than during
function analyze_weighted_profile_discrepancy(loss, forward_map_output::Dict, data, target)
    total_discrepancy = zero(eltype(data.grid))
    field_names = Tuple(loss.fields)

    for (field_index, field_name) in enumerate(field_names)
        # model_field = getproperty(forward_map_output, field_name)[target]
        model_field = forward_map_output[field_name][target]
        data_field = getproperty(data, field_name)[target]

        # Calculate the per-field profile-based disrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)
        if target == 1
            println(field_discrepancy)
        end

        # Accumulate weighted profile-based disrepancies in the total discrepancyor
        total_discrepancy += get_weight(loss.weights, field_index) * field_discrepancy # accumulate discrepancyor
    end

    return nan2inf(total_discrepancy)
end

# Evaluate loss function AFTER the forward map rather than during
function (loss::LossFunction)(forward_map_output::Dict, data)
    # @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, forward_map_output, data, 1)
    # @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, forward_map_output, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, forward_map_output, data, target)
    end
    # println(loss.time_series.data[1])

    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end
