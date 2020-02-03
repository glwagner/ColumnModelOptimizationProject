module ModularKPPOptimization

export
    DefaultFreeParameters,
    DefaultStdFreeParameters,
    BasicParameters,
    WindMixingParameters,
    WindMixingAndShapeParameters,
    WindMixingAndExponentialShapeParameters,
    WindyConvectionParameters

using OceanTurb, PyPlot

using ..ColumnModelOptimizationProject

import ColumnModelOptimizationProject: set!

parameter_latex_guide = Dict(
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

function set!(cm::ColumnModelOptimizationProject.ColumnModel{<:ModularKPP.Model}, 
              cd::ColumnData, i)

    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])

    cd.S != nothing && set!(cm.model.solution.S, cd.S[i])

    cm.model.clock.time = cd.t[i]
    return nothing
end

include("modular_kpp_parameter_sets.jl")
include("modular_kpp_models.jl")
include("modular_kpp_visualization.jl")

end # module
