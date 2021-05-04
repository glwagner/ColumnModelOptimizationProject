module TKEMassFluxOptimization

export parameter_latex_guide

using OceanTurb, PyPlot

using ..ColumnModelOptimizationProject

import ColumnModelOptimizationProject: set!

parameter_latex_guide = Dict(
     :Cᴰ   => L"C^D",
     :Cᴷu  => L"C^K_U",
     :Cᴷc  => L"C^K_C",
     :Cᴷe  => L"C^K_E",
     :CᴷPr => L"C^K_\mathrm{Pr}",
     :Cᴸᵟ  => L"C^{\ell}_\delta",
     :Cᴸʷ  => L"C^{\ell}_w",
     :Cᴸᵇ  => L"C^{\ell}_b",
     :Cʷu★ => L"C^w_{u\star}",
     :CʷwΔ => L"C^ww\Delta",
     :Cᴬ => L"C^A",
     :CᴷRiʷ => L"C^KRi^w",
     :CᴷRiᶜ => L"C^KRi^c",
     :Cᴷu⁻ => L"C^Kc^-",
     :Cᴷu⁺ => L"C^Kc^+",
     :Cᴷc⁻ => L"C^Kc^-",
     :Cᴷc⁺ => L"C^Kc^+",
     :Cᴷe⁻ => L"C^Ke^-",
     :Cᴷe⁺ => L"C^Ke^+",
     :Cᴬu => L"C^A_U",
     :Cᴬc => L"C^A_C",
     :Cᴬe => L"C^A_E",
)

# parameter_latex_guide = Dict(
#      :Cᴰ   => L"C^D",
#      :Cᴷu  => L"C^K_u",
#      :Cᴷc  => L"C^K_c",
#      :Cᴷe  => L"C^K_e",
#      :CᴷPr => L"C^K_\mathrm{Pr}",
#      :Cᴸᵟ  => L"C^\ell_\delta",
#      :Cᴸʷ  => L"C^\ell_w",
#      :Cᴸᵇ  => L"C^\ell_b",
#      :Cʷu★ => L"C^w_{e}",
#      :CʷwΔ => L"C^ww\Delta",
#      :Cᴬ => L"C^A",
#      :CᴷRiʷ => L"C^KRi^w",
#      :CᴷRiᶜ => L"C^KRi^c",
#      :Cᴷc⁻ => L"C^Kc^-",
#      :Cᴷc⁺ => L"C^Kc^+",
#      :Cᴷe⁻ => L"C^Ke^-",
#      :Cᴷe⁺ => L"C^Ke^+",
#      :Cᴬu => L"C^A_u",
#      :Cᴬc => L"C^A_c",
#      :Cᴬe => L"C^A_e",
# )

@free_parameters WindMixingParameters Cᴸʷ Cᴸᵇ Cᴰ Cᴷᵤ Cᴾʳ Cᴷₑ Cʷu★

#
# Basic functionality
#

function set!(cm::ColumnModelOptimizationProject.ColumnModel{<:TKEMassFlux.Model},
              cd::ColumnData, i)

    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.e, cd.e[i])

    if cd.S === nothing
        set!(cm.model.solution.S, 0)
    else
        set!(cm.model.solution.S, cd.S[i])
    end

    cm.model.clock.time = cd.t[i]

    return nothing
end

include("tke_mass_flux_models.jl")

end # module
