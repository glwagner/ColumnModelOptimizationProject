module ColumnModelOptimizationProject

export
    dictify,

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
    ColumnModel,

    # visualization.jl
    visualize_targets,
    visualize_realization,

    # models/kpp_optimization.jl
    KPPOptimization

using
    StaticArrays,
    OceanTurb,
    JLD2,
    PyCall,
    Printf,
    PyPlot

import OceanTurb: set!, absolute_error

dictify(p) = Dict((k, getproperty(p, k)) for k in propertynames(p))

set!(::Nothing, args...) = nothing # placeholder

include("file_wrangling.jl")
include("data_analysis.jl")
include("column_models.jl")
include("loss_functions.jl")
include("visualization.jl")

include("models/kpp_optimization.jl")

end # module
