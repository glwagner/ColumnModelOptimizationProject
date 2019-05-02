module KPPOptimization

export
    DefaultFreeParameters,
    DefaultStdFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,

    temperature_cost,
    weighted_cost,
    simple_flux_model

using
    ColumnModelOptimizationProject,
    OceanTurb,
    StaticArrays

import Base: similar
import ColumnModelOptimizationProject: ColumnModel
import OceanTurb: set!

#
# Basic functionality
#

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

function similar(p::FreeParameters{N, T}) where {N, T}
    return eval(Expr(:call, typeof(p), (zero(T) for i = 1:N)...))
end

function set!(cm::ColumnModel{<:KPP.Model}, params::FreeParameters)
    cm.model.parameters = KPP.Parameters(; dictify(params)...)
    return nothing
end

function set!(cm::ColumnModel{<:KPP.Model}, cd::ColumnData, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.S, cd.S[i])
    cm.model.clock.time = cd.t[i]
    return nothing
end

#
# Parameter sets
#

Base.@kwdef mutable struct ShearNeutralParameters{T} <: FreeParameters{3, T}
    CSL :: T  # Surface layer fraction
    CRi :: T  # Critical bulk Richardson number
    Cτ  :: T  # Von Karman constant
end

Base.@kwdef mutable struct FreeConvectionParameters{T} <: FreeParameters{4, T}
     CSL :: T  # Surface layer fraction
     CNL :: T
     CKE :: T
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

function DefaultStdFreeParameters(relative_std, freeparamtype)
    allparams = KPP.Parameters()
    param_stds = (relative_std * getproperty(allparams, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, freeparamtype, param_stds...))
end

function ColumnModel(cd::ColumnData, Δt; kwargs...)
    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu, Bz=cd.Bz, kwargs...)
    return ColumnModel(model, Δt)
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
