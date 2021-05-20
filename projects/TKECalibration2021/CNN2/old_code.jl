"""
For each field variable ϕ ∈ {U, V, T, E}, we execute the following procedure:
    For each training simulation that involves that field
        For each time step i ∈ 1:Nt-60 in the simulation
            run line-search gradient descent on α_ϕ, α_ϕ starting from [α_ϕ, α_ϕ]=[0.5,0.5],
                where the loss at each iteration of the descent is computed by
                (1) initializing the CATKE model to the LES profiles at that time step (ϕ_CATKE[0] ← ϕ_LES[i])
                (2) evolving CATKE forward for 6 time steps (corresponding to 1 hour in simulation time because Δt=10 mins)
                (3) evaluating (1/64)||ϕ_CATKE[6] - ϕ_LES[i+6]||^2, the MSE between the CATKE prediction for ϕ and the LES solution for ϕ.
"""

# data_for_field = Dict(:U => [], :T => [], :V => [], :e => [])
# for LEScase in values(FourDaySuite)
#     fields = !(LEScase.stressed) ? (:T, :e) :
#              !(LEScase.rotating) ? (:T, :U, :e) :
#                                    (:T, :U, :V, :e)
#     for field in fields
#         push!(data_for_field[field], ColumnData(LEScase.filename))
#     end
# end

# for field in (:U, :V, :T)
#
#     for data in data_for_field[field]
#
#         for start_index = 1:length(data)-Nt
#
#             get_optimal_α_σ(model, data, field, start_index)
#
# end

# function get_normalization_functions(LESdata)
#     normalize_function = Dict()
#     for field in (:U, :V, :T)
#
#         μs = []
#         σs = []
#         for LEScase in values(LESdata)
#             data = ColumnData(LEScase.filename)
#             fields = !(LEScase.stressed) ? (:T, :e) :
#                      !(LEScase.rotating) ? (:T, :U, :e) :
#                                            (:T, :U, :V, :e)
#
#             first = LEScase.first
#             last = LEScase.last == nothing ? length(data) : LEScase.last
#             targets = (first, last)
#
#             if field in fields
#                 push!(μs, profile_mean(data, field, targets))
#                 push!(σs, sqrt(max_variance(data, field, targets)))
#
#             end
#         end
#         μ = mean(μs)
#         σ = mean(σs)
#         normalize(Φ) = (Φ .- μ) ./ σ
#         normalize_function[field] = normalize
#     end
#
#     return normalize_function
# end
# normalize_function = get_normalization_functions(FourDaySuite)

function full_time_series(parameters::Vector, column_model, column_data, fields, weights, start_index)

    initialize_forward_run!(column_model, column_data, parameters[start_index], start_index)

    grid = column_model.grid

    output = ModelTimeSeries([CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt],
                             [CellField(grid) for i = 1:Nt])

    U = column_model.solution.U
    V = column_model.solution.V
    T = column_model.solution.T
    e = column_model.solution.e

    Nt = length(data.t) - start + 1

    for i in 1:Nt

        set!(column_model, parameters[i])

        # Simulation time step
        target = i + start_index - 1

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
            # total_discrepancies[field_index] += weights[field_index] * mean(discrepancy)
        end
    end

    # return nan2inf(total_discrepancy)
    return output
end
