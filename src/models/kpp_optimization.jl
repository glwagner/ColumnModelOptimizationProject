module KPPOptimization

export
    DefaultFreeParameters,
    DefaultStdFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,
    BasicParameters,

    set!,

    smoothstep,
    simple_flux_model,
    visualize_model,
    save_data,
    init_data,
    generate_data

using
    ColumnModelOptimizationProject,
    OceanTurb,
    OceanTurb.Plotting,
    StaticArrays,
    PyPlot,
    JLD2

import Base: similar
import ColumnModelOptimizationProject: ColumnModel
import OceanTurb: set!

latexparams = Dict(
      :CRi => L"C^\mathrm{Ri}",
      :CKE => L"C^\mathcal{E}",
      :CNL => L"C^{NL}",
      :Cτ  => L"C^\tau",
    :Cstab => L"C^\mathrm{stab}",
    :Cunst => L"C^\mathrm{unst}",
     :Cb_U => L"C^b_U",
     :Cb_T => L"C^b_T",
     :Cd_U => L"C^d_U",
     :Cd_T => L"C^d_T"
)

include("kpp_utils.jl")

#
# Basic functionality
#

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

Base.@kwdef mutable struct BasicParameters{T} <: FreeParameters{8, T}
    CRi   :: T # Critical bulk Richardson number
    CKE   :: T # Unresolved kinetic energy constant
    CNL   :: T # Non-local flux constant
    Cτ    :: T # Von Karman constant
    Cstab :: T # Stable buoyancy flux parameter for wind-driven turbulence
    Cunst :: T # Unstable buoyancy flux parameter for wind-driven turbulence
    Cb_U  :: T # Buoyancy flux parameter for convective turbulence
    Cb_T  :: T # Buoyancy flux parameter for convective turbulence
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

end # module
