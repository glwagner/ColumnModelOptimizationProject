using JLD2, BSON
using TKECalibration2021
using ColumnModelOptimizationProject
using ColumnModelOptimizationProject: TKEMassFluxOptimization, TKEMassFlux,
                                      run_until!, mean_variance, nan2inf, set!
using ColumnModelOptimizationProject.TKEMassFluxOptimization: set!

using OceanTurb: CellField
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using Distributions, Random, LinearAlgebra, Optim, Plots
using ProgressMeter: @showprogress
using BSON

files =  ["free_convection",
          "strong_wind",
          "strong_wind_no_rotation",
          "weak_wind_strong_cooling",
          "strong_wind_weak_cooling",
          ]

JLD2.@load "CNN/training_pairs.jld2" xtrain ytrain
BSON.@load "./runs/model.bson" model
@load "runs/model.bson" model

include("./GaussianProcess/whole.jl")
include("generate_training_pairs_utils.jl")

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

@free_parameters(CATKEparameters,
                Cᴬu, Cᴬc, Cᴬe,
                Cᴷu, Cᴷc, Cᴷe)

profile_indices = 25:64
data_indices = 50:127
starti = 1

function get_timeseries(training_simulations; Δt = 10.0)

    Truth = Dict()
    CATKE_CNN = Dict()
    CATKE_GP = Dict()
    CATKE_default = Dict()

    relative_weights = Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 0.0)
    defaults = TKEParametersRiDependent([0.7213, 0.7588, 0.1513, 0.7337, 0.3977, 1.7738, 0.1334, 1.2247, 2.9079, 1.1612, 3.618765767212402, 1.3051900286568907])

    for LEScase in values(training_simulations)
      fields = !(LEScase.stressed) ? (:T, :e) :
               !(LEScase.rotating) ? (:T, :U, :e) :
                                     (:T, :U, :V, :e)

      column_data = ColumnData(LEScase.filename)

      column_model = TKEMassFluxOptimization.ColumnModel(column_data, Δt,
                                                            N = 64,
                                                 mixing_length = TKEMassFlux.SimpleMixingLength(),
                                                tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                                            eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                                                  tke_equation = TKEMassFlux.TKEParameters(),
                                         convective_adjustment = TKEMassFlux.VariablePrandtlConvectiveAdjustment(),
                                           )

       rw = [relative_weights[f] for f in fields]
       weights = estimate_weights(column_data, rw, fields)

       # Set model to custom defaults
       set!(column_model, defaults)

       default_model(UVTE) = [0.1023, 1.3936, 0.3576, 0.1723, 0.0676, 0.6067]

       # cnn_model(UVTE) = model(reshape(UVTE, 160, 1, 1))

       gp_model = get_gp_model(xtrain, ytrain, SquaredExponentialI(10^(-16.0), 1.0, euclidean_distance))
       # a = mean(ytrain, dims=1)
       # gp_model(UVTE) = [a[1]...]
       # gp_model(UVTE) = [1.0,1.0,1.01,1.0,1.0,1.0]


       initial_parameters = custom_defaults(column_model, CATKEparameters)

       generate_timeseries(model) = full_time_series(model, CATKEparameters, column_model, column_data,
                                                    fields, weights, starti, get_UVTE, initial_parameters)

       # cnn_timeseries, cnn_loss     = generate_timeseries(cnn_model)
       gp_timeseries, gp_loss       = generate_timeseries(gp_model)
       default_timeseries, def_loss = generate_timeseries(default_model)

       Truth[column_data.name]         = column_data
       # CATKE_CNN[column_data.name]     = cnn_timeseries
       CATKE_GP[column_data.name]      = gp_timeseries
       CATKE_default[column_data.name] = default_timeseries

       # println(cnn_loss, gp_loss, def_loss)
       println(gp_loss, def_loss)
       println("$gp_loss $def_loss")

    end
    return Truth, CATKE_CNN, CATKE_GP, CATKE_default
end

Truth, CATKE_CNN, CATKE_GP, CATKE_default = get_timeseries(SixDaySuite; Δt = 10.0)

f1 = Dict(
    "u" => 𝒟 -> 𝒟.U, "v" => 𝒟 -> 𝒟.V, "e" => 𝒟 -> 𝒟.e, "T"  => 𝒟 -> 𝒟.T
)

f2 = Dict(
    "u" => 𝒟 -> 𝒟.U, "v" => 𝒟 -> 𝒟.V, "E" => 𝒟 -> 𝒟.E, "T"  => 𝒟 -> 𝒟.T
)

x_lims = Dict(
    "u" => (-0.3,0.4), "v" => (-0.3,0.1), "T" => (19.6,20.0), "e" => (-0.5,4),
)

function plot_(p, name, file, time, DataDict, label_, f; lw_ = 3, la_ = 1.0)
    field = f[name](DataDict[file])
    Nz = field[1].grid.N
    z = parent(field[1].grid.zc[1:Nz-1])

    profile = field[time].data[1:Nz-1]
    if !any(i -> isnan(i), profile)
        plot!(profile, z, color=colors[name], linewidth=lw_, la=la_, label=label_, xlims=x_lims[name], legend=false)
    end
    if name=="u"
        profile = f["v"](DataDict[file])[time].data[1:Nz-1]
        if !any(i -> isnan(i), profile)
            plot!(profile, z, color=colors["v"], linewidth=lw_, la=la_, xlims=x_lims[name], legend=false)
        end
    end
    p
end

function stacked_(file, time)
    u = Plots.plot(plot_titlefontsize=20, legend=:outerright)
    plot_(u, "u", file, time+starti, Truth, "Truth", f1; lw_ = 10, la_ = 0.3)
    plot_(u, "u", file, time, CATKE_GP, "Prediction", f2; lw_ = 3, la_ = 1.0)
    t = Plots.plot(plot_titlefontsize=20, legend=:outerright)
    plot_(t, "T", file, time+starti, Truth, "Truth", f1; lw_ = 10, la_ = 0.3)
    plot_(t, "T", file, time, CATKE_GP, "Prediction", f2; lw_ = 3, la_ = 1.0)
    layout=@layout[a; b]
    p = Plots.plot(t, u, layout=layout)
    plot!(tickfontsize=20, ylims=(-256,0), ticks=false)
    plot!(widen=true, grid=false, framestyle=:none)
    return p
end

function whole_suite_animation(time)
    a = stacked_("free_convection", time)
    b = stacked_("strong_wind", time)
    c = stacked_("strong_wind_no_rotation", time)
    d = stacked_("weak_wind_strong_cooling", time)
    e = stacked_("strong_wind_weak_cooling", time)
    layout = @layout [a b c d e]
    p = Plots.plot(a, b, c, d, e, layout=layout, framestyle=:none)
    plot!(bottom_margin=0*Plots.mm, size=(1800, 800))
    return p
end

# whole_suite_animation(300)

anim = @animate for time=1:2:571
    p = whole_suite_animation(time)
end
Plots.gif(anim, "./bad.gif", fps=400)
visualize_and_save(ce, params, "CNN/")
