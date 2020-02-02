module TKEMassFluxOptimization

export
    WindMixingParameters,
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

include("tke_mass_flux_utils.jl")

#
# Basic functionality
#

function set!(cm::ColumnModel{<:ModularKPP.Model}, cd::ColumnData, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.e, cd.e[i])

    try
        set!(cm.model.solution.S, cd.S[i])
    catch end

    cm.model.clock.time = cd.t[i]

    return nothing
end

#
# Parameter sets
#

Base.@kwdef mutable struct WindMixingParameters{T} <: FreeParameters{7, T}
     CLz :: T
     CLb :: T
     CLΔ :: T
     CDe :: T
    CK_U :: T
    CK_T :: T
    CK_e :: T
end

Base.similar(p::WindMixingParameters{T}) where T = 
    WindMixingParameters{T}(0, 0, 0, 0, 0, 0, 0)

end # module
