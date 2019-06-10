module ColumnModelOptimizationProject

export
    dictify,
    FreeParameters,
    get_free_parameters,

    # file_wrangling.jl
    iterations,
    times,
    getdata,
    getconstant,
    getbc,
    getic,
    getgridparams,
    getdataparams,

    # data_analysis.jl
    removespines,
    summarize_data,

    # column_models.jl
    ColumnData,
    target_times,
    initial_time,
    ColumnModel,

    # visualization.jl
    visualize_targets,
    visualize_realization,

    # loss_functions.jl
    temperature_loss,
    velocity_loss,
    weighted_fields_loss,

    # models/kpp_optimization.jl
    #KPPOptimization,
    ModularKPPOptimization

using
    StaticArrays,
    OceanTurb,
    JLD2,
    Printf

include( joinpath(pathof(OceanTurb), "..", "..", "plotting", "pyplot_utils.jl") )

using PyPlot, PyCall, .OceanTurbPyPlotUtils

import OceanTurb: set!, absolute_error

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

function similar(p::FreeParameters{N, T}) where {N, T}
    return eval(Expr(:call, typeof(p), (zero(T) for i = 1:N)...))
end

dictify(p) = Dict((k, getproperty(p, k)) for k in propertynames(p))

set!(::Nothing, args...) = nothing # placeholder

function get_free_parameters(cm)
    paramnames = Dict()
    paramtypes = Dict()
    for pname in propertynames(cm.model)
        p = getproperty(cm.model, pname)
        if typeof(p) <: OceanTurb.AbstractParameters
            paramnames[pname] = propertynames(p)
            paramtypes[pname] = typeof(p)
        end
    end
    return paramnames, paramtypes
end

include("file_wrangling.jl")
include("data_analysis.jl")
include("column_models.jl")
include("loss_functions.jl")
include("visualization.jl")

#include("models/kpp_optimization.jl")
include("models/modular_kpp_optimization.jl")

end # module
