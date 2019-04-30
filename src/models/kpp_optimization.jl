module KPPOptimization

export
    DefaultFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,

    temperature_based_nll,
    simple_flux_model,
    compare_with_data,
    visualize_compare_with_data

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

mutable struct SensitiveParameters{T} <: FreeParameters{4, T}
     CRi :: T
     CNL :: T
     CKE :: T  # Unresolved turbulence parameter
end

mutable struct ShearNeutralParameters{T} <: FreeParameters{6, T}
    CSL :: T  # Surface layer fraction
    CNL :: T  # Non-local flux proportionality constant
    CRi :: T  # Critical bulk Richardson number
    Cτ  :: T  # Von Karman constant
    KU₀ :: T  # Background diffusivity for U
    KT₀ :: T  # Background diffusivity for T
end

mutable struct FreeConvectionParameters{T} <: FreeParameters{6, T}
     CNL :: T
     CKE :: T
      Cτ :: T
    Cb_U :: T
    Cb_T :: T
      K₀ :: T
end

mutable struct ShearUnstableParameters{T} <: FreeParameters{12, T}
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
    K₀    :: T  # Background diffusivity
end

function DefaultFreeParameters(freeparamtype)
    allparams = KPP.Parameters()
    freeparams = (getproperty(allparams, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, freeparamtype, freeparams...))
end

function ColumnModel(cd::ColumnData, Δt; kwargs...)
    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu, kwargs...)
    return ColumnModel(model, Δt)
end

function set!(cm::ColumnModel{<:KPP.Model}, cd, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.S, cd.S[i])
    return nothing
end


#
# Negative log likelihood functions
#

function temperature_based_nll(params, column_model, column_data)

    # Initialize the model
    kpp_parameters = KPP.Parameters(; dictify(params)...)
    column_model.model.parameters = kpp_parameters

    set!(column_model, column_data, column_data.i_initial)
    column_model.model.clock.time = column_data.t[column_data.i_initial]
    column_model.model.clock.iter = 0

    err = 0.0
    for i in column_data.i_compare
        run_until!(column_model.model, column_model.Δt, column_data.t[i])
        err += relative_error(column_model.model.solution.T, column_data.T[i]) / length(column_data.i_compare)
    end

    return err
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
function simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                           parameters=KPP.Parameters())

    model = KPP.Model(N=N, L=L, parameters=parameters, constants=constants,
                      stepper=:BackwardEuler)

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
