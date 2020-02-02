module ModularKPPOptimization

export
    DefaultFreeParameters,
    DefaultStdFreeParameters,
    BasicParameters,
    WindMixingParameters,
    WindMixingAndShapeParameters,
    WindMixingAndExponentialShapeParameters,
    WindyConvectionParameters,

    simple_flux_model,
    latexparams

using
    OceanTurb,
    StaticArrays,
    PyPlot,
    JLD2

using ..ColumnModelOptimizationProject

import Base: similar
import ColumnModelOptimizationProject: ColumnModel, set!

latexparams = Dict(
      :CRi => L"C^\mathrm{Ri}",
      :CSL => L"C^\mathrm{SL}",
      :CKE => L"C^\mathcal{E}",
      :CNL => L"C^{NL}",
      :Cτ  => L"C^\tau",
    :Cstab => L"C^\mathrm{stab}",
    :Cunst => L"C^\mathrm{unst}",
     :Cb_U => L"C^b_U",
     :Cb_T => L"C^b_T",
     :Cd_U => L"C^d_U",
     :Cd_T => L"C^d_T",
      :CS0 => L"C^{S_0}",
      :CS1 => L"C^{S_1}",
      :CSe => L"C^{S_e}",
      :CSd => L"C^{S_d}"
)

include("modular_kpp_utils.jl")

#
# Basic functionality
#

function set!(cm::ColumnModel{<:ModularKPP.Model}, cd::ColumnData, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    #set!(cm.model.solution.S, cd.S[i])

    cm.model.clock.time = cd.t[i]
    return nothing
end

#
# Parameter sets
#

Base.@kwdef mutable struct BasicParameters{T} <: FreeParameters{9, T}
      CRi :: T
      CSL :: T
      CKE :: T
      CNL :: T
      Cτ  :: T
    Cstab :: T
    Cunst :: T
     Cb_U :: T
     Cb_T :: T
end

Base.similar(p::BasicParameters{T}) where T = BasicParameters{T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

Base.@kwdef mutable struct WindyConvectionParameters{T} <: FreeParameters{15, T}
      CRi :: T
      CSL :: T
     CKSL :: T
      CKE :: T
      CNL :: T
      Cτ  :: T
    Cunst :: T
     Cb_U :: T
     Cb_T :: T
    Cmτ_T :: T
    Cmτ_U :: T
    Cmb_T :: T
    Cmb_U :: T
     Cd_U :: T
     Cd_T :: T
end

Base.similar(p::WindyConvectionParameters{T}) where T =
    WindyConvectionParameters{T}((0 for i = 1:length(fieldnames(WindyConvectionParameters)))...)

Base.@kwdef mutable struct WindMixingParameters{T} <: FreeParameters{3, T}
      CRi :: T
      CSL :: T
      Cτ  :: T
end

Base.similar(p::WindMixingParameters{T}) where T = WindMixingParameters{T}(0, 0, 0)

Base.@kwdef mutable struct WindMixingAndShapeParameters{T} <: FreeParameters{5, T}
      CRi :: T
      CSL :: T
      Cτ  :: T
      CS0 :: T
      CS1 :: T
end

Base.similar(p::WindMixingAndShapeParameters{T}) where T =
    WindMixingAndShapeParameters{T}(0, 0, 0, 0, 0)

Base.@kwdef mutable struct WindMixingAndExponentialShapeParameters{T} <: FreeParameters{6, T}
      CRi :: T
      CSL :: T
      Cτ  :: T
      CS0 :: T
      CSe :: T
      CSd :: T
end

Base.similar(p::WindMixingAndExponentialShapeParameters{T}) where T =
    WindMixingAndExponentialShapeParameters{T}(0, 0, 0, 0, 0, 0)

function DefaultStdFreeParameters(relative_std, freeparamtype)
    allparams = KPP.Parameters()
    param_stds = (relative_std * getproperty(allparams, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, freeparamtype, param_stds...))
end

end # module
