module TKEMassFluxOptimization

export
    WindMixingParameters

using OceanTurb, PyPlot

using ..ColumnModelOptimizationProject

import ColumnModelOptimizationProject: set!

#
# Basic functionality
#

function set!(cm::ColumnModelOptimizationProject.ColumnModel{<:TKEMassFlux.Model}, 
              cd::ColumnData, i)

    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    set!(cm.model.solution.e, cd.e[i])

    cd.S != nothing && set!(cm.model.solution.S, cd.S[i])

    cm.model.clock.time = cd.t[i]

    return nothing
end

include("tke_mass_flux_models.jl")
include("tke_mass_flux_parameters.jl")

end # module
