# Evaluate loss function AFTER the forward map rather than during
mutable struct ModelTimeSeries{UU, VV, TΘ, SS, EE}
                     U :: UU
                     V :: VV
                     T :: TΘ
                     S :: SS
                     e :: EE
end

# function model_time_series(parameters, model_plus_Δt, data, loss) # loss is needed for targets -- that's all
#
#     initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])
#
#     grid = model_plus_Δt.grid
#
#     output = ForwardMapOutput([CellField(grid) for i = 1:length(loss.targets)],
#                               [CellField(grid) for i = 1:length(loss.targets)],
#                               [CellField(grid) for i = 1:length(loss.targets)],
#                               [CellField(grid) for i = 1:length(loss.targets)],
#                               [CellField(grid) for i = 1:length(loss.targets)])
#
#     U = model_plus_Δt.solution.U
#     V = model_plus_Δt.solution.V
#     T = model_plus_Δt.solution.T
#     S = model_plus_Δt.solution.S
#     e = model_plus_Δt.solution.e
#
#     for (i, target) in enumerate(loss.targets)
#         run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[target])
#
#         U_snapshot = output.U[i].data
#         V_snapshot = output.V[i].data
#         T_snapshot = output.T[i].data
#         S_snapshot = output.S[i].data
#         e_snapshot = output.e[i].data
#
#         U_snapshot .= U.data
#         V_snapshot .= V.data
#         T_snapshot .= T.data
#         S_snapshot .= S.data
#         e_snapshot .= e.data
#     end
#
#     return output
# end

function model_time_series(parameters, model_plus_Δt, data)

    Nt = length(data.t)

    initialize_forward_run!(model_plus_Δt, data, parameters, 1)

    grid = model_plus_Δt.grid

    output = ForwardMapOutput([CellField(grid) for i = 1:Nt],
                              [CellField(grid) for i = 1:Nt],
                              [CellField(grid) for i = 1:Nt],
                              [CellField(grid) for i = 1:Nt],
                              [CellField(grid) for i = 1:Nt])

    U = model_plus_Δt.solution.U
    V = model_plus_Δt.solution.V
    T = model_plus_Δt.solution.T
    S = model_plus_Δt.solution.S
    e = model_plus_Δt.solution.e

    for i in 1:Nt
        run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[i])

        U_snapshot = output.U[i].data
        V_snapshot = output.V[i].data
        T_snapshot = output.T[i].data
        S_snapshot = output.S[i].data
        e_snapshot = output.e[i].data

        U_snapshot .= U.data
        V_snapshot .= V.data
        T_snapshot .= T.data
        S_snapshot .= S.data
        e_snapshot .= e.data
    end

    return output
end

# Evaluate loss function AFTER the forward map rather than during
function analyze_weighted_profile_discrepancy(loss, forward_map_output::ModelTimeSeries, data, target)
    total_discrepancy = zero(eltype(data.grid))
    field_names = Tuple(loss.fields)

    # target = loss.targets[index]

    for (field_index, field_name) in enumerate(field_names)
        model_field = getproperty(forward_map_output, field_name)[target]
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
function (loss::LossFunction)(forward_map_output::ModelTimeSeries, data)
    @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, forward_map_output, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, forward_map_output, data, target)
    end

    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end
