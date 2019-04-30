module KPPOptimization

export
    DefaultFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,

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
# Loss functions
#

"""
    temperature_loss(params, model, data; iters=1)

Compute the error between model and data for one iteration of
the `model.turbmodel`.
"""
function temperature_loss(params, model, data; iters=1)
    kpp_parameters = KPP_Parameters(params, data.K₀)
    turbmodel = model.turbmodel
    turbmodel.parameters = kpp_parameters

    # Set initial condition as first non-trivial time-step
    turbmodel.solution.U = data.U[2]
    turbmodel.solution.V = data.V[2]
    turbmodel.solution.T = data.T[2]

    loss = 0.0

    for i = 3:3+iters
        run_until!(turbmodel, model.dt, data.t[i])
        T_model = OceanTurb.data(turbmodel.solution.T)
        loss += mean(data.T[i].^2 .- T_model.^2)
    end

    return loss
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
