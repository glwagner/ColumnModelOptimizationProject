
function full_time_series(param_model, CATKEParameters, column_model, column_data, fields, weights, start_index, get_UVTE, initial_parameters)

    initialize_forward_run!(column_model, column_data, initial_parameters, start_index)

    grid = column_model.grid

    Nt = length(column_data.t) - 1

    output = ModelTimeSeries([CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt])

    U = column_model.solution.U
    V = column_model.solution.V
    T = column_model.solution.T
    e = column_model.solution.e

    # Set snapshot to empty CellField
    U_snapshot = output.U[1].data
    V_snapshot = output.V[1].data
    T_snapshot = output.T[1].data
    e_snapshot = output.e[1].data

    total_discrepancy = zero(eltype(column_model.grid))
    coarse_data = discrepancy = CellField(grid)

    for i in 1:Nt

        UVTE = get_UVTE([U_snapshot...], [V_snapshot...], [T_snapshot...], [e_snapshot...])

        parameters = param_model(UVTE)

        set!(column_model, CATKEparameters(parameters))

        # Simulation time step
        target = i + 1

        # Evolve model for Nt timesteps
        run_until!(column_model.model, column_model.Δt, column_data.t[target])

        # Set snapshot to empty CellField
        U_snapshot = output.U[i].data
        V_snapshot = output.V[i].data
        T_snapshot = output.T[i].data
        e_snapshot = output.e[i].data

        # Fill empty CellField with model data
        U_snapshot .= U.data
        V_snapshot .= V.data
        T_snapshot .= T.data
        e_snapshot .= e.data

        for (field_index, field_name) in enumerate(fields)

            model_field = getproperty(column_model.solution, field_name)
            data_field = getproperty(column_data, field_name)[target]

            set!(coarse_data, data_field)

            for i in eachindex(discrepancy)
                @inbounds discrepancy[i] = (coarse_data[i] - model_field[i])^2
            end

            total_discrepancy += weights[field_index] * mean(discrepancy)
        end
    end

    return output, nan2inf(total_discrepancy / Nt)
end
