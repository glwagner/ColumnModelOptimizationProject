module KPPOptimization

export
    DefaultFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,

    temperature_cost,
    weighted_cost,
    simple_flux_model,
    compare_with_data,
    visualize_compare_with_data,
    visualize_target,
    visualize_realization

using
    ColumnModelOptimizationProject,
    OceanTurb,
    StaticArrays,
    JLD2,
    Printf,
    PyPlot

import Base: similar
import PyCall: pyimport
import ColumnModelOptimizationProject: ColumnModel
import OceanTurb: set!

include("kpp_visualization.jl")

#
# Parameter sets
#

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

function similar(p::FreeParameters{N, T}) where {N, T}
    return eval(Expr(:call, typeof(p), (zero(T) for i = 1:N)...))
end

Base.@kwdef mutable struct ShearNeutralParameters{T} <: FreeParameters{3, T}
    CSL :: T  # Surface layer fraction
    CRi :: T  # Critical bulk Richardson number
    Cτ  :: T  # Von Karman constant
end

Base.@kwdef mutable struct FreeConvectionParameters{T} <: FreeParameters{5, T}
     CNL :: T
     CKE :: T
      Cτ :: T
    Cb_U :: T
    Cb_T :: T
end

Base.@kwdef mutable struct ShearUnstableParameters{T} <: FreeParameters{12, T}
    CSL   :: T  # Surface layer fraction
    Cτ    :: T  # Von Karman constant
    CNL   :: T  # Non-local flux proportionality constant
    Cunst :: T  # Unstable buoyancy flux parameter for wind-driven turbulence
    Cb_U  :: T  # Buoyancy flux parameter for convective turbulence
    Cτb_U :: T  # Wind stress parameter for convective turbulence
    Cb_T  :: T  # Buoyancy flux parameter for convective turbulence
    Cτb_T :: T  # Wind stress parameter for convective turbulence
    Cd_U  :: T  # Wind mixing regime threshold for momentum
    Cd_T  :: T  # Wind mixing regime threshold for tracers
    CRi   :: T  # Critical bulk Richardson number
    CKE   :: T  # Unresolved turbulence parameter
end

function DefaultFreeParameters(freeparamtype)
    allparams = KPP.Parameters()
    freeparams = (getproperty(allparams, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, freeparamtype, freeparams...))
end

function ColumnModel(cd::ColumnData, Δt; kwargs...)
    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu, Bz=cd.Bz, kwargs...)
    return ColumnModel(model, Δt)
end

function set!(cm::ColumnModel{<:KPP.Model}, cd, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.S, cd.S[i])
    cm.model.clock.time = cd.t[i]
    return nothing
end

#
# Negative log likelihood functions
#

function temperature_cost(params, column_model, column_data)

    # Initialize the model
    kpp_parameters = KPP.Parameters(; dictify(params)...)
    column_model.model.parameters = kpp_parameters

    set!(column_model, column_data, column_data.initial)

    err = zero(eltype(column_model.model.solution.U))
    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])
        err += absolute_error(column_model.model.solution.T, column_data.T[i])
    end

    if isnan(err)
        err = Inf
    end

    return err / length(column_data.targets)
end

function weighted_cost(params, column_model, column_data, weights)

    # Initialize the model
    kpp_parameters = KPP.Parameters(; KU₀=column_data.ν, KT₀=column_data.κ, KS₀=column_data.κ,
                                      dictify(params)...)
    column_model.model.parameters = kpp_parameters

    set!(column_model, column_data, column_data.initial)

    fields = (:U, :V, :T, :S)
    total_err = zero(eltype(column_model.model.solution.U))

    for i in column_data.targets
        run_until!(column_model.model, column_model.Δt, column_data.t[i])

        for (j, fld) in enumerate(fields)
            field_err = absolute_error(
                getproperty(column_model.model.solution, fld),
                getproperty(column_data, fld)[i])

            # accumulate error
            total_err += weights[j] * field_err
        end
    end

    if isnan(total_err)
        total_err = Inf
    end

    return total_err / length(column_data.targets)
end


#
# Models
#

"""
    simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                      parameters=KPP.Parameters())

Construct a model forced by 'simple', constant atmospheric buoyancy flux `Fb`
and velocity flux `Fu`, with resolution `N`, domain size `L`, and
and initial linear buoyancy gradient `Bz`.
"""
function simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0, parameters=KPP.Parameters())

    model = KPP.Model(N=N, L=L, parameters=parameters, constants=constants, stepper=:BackwardEuler)

    # Initial condition
    Tz = model.constants.α * model.constants.g * Bz
    T₀(z) = 20 + Tz*z
    model.solution.T = T₀

    # Fluxes
    Fθ = Fb / (model.constants.α * model.constants.g)
    model.bcs.U.top = FluxBoundaryCondition(Fu)
    model.bcs.T.top = FluxBoundaryCondition(Fθ)
    model.bcs.T.bottom = GradientBoundaryCondition(Tz)

    return model
end

function simple_flux_model(datapath::AbstractString; N=nothing)
    data_params, constants_dict = getdataparams(datapath)
    constants = KPP.Constants(; constants_dict...)
    if N != nothing
        data_params[:N] = N
    end
    simple_flux_model(constants; data_params...)
end

end # module
