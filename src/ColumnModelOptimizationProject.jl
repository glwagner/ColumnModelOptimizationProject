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

    KPPOptimization

using
    StaticArrays,
    OceanTurb,
    JLD2,
    PyCall,
    Printf,
    PyPlot

import Base: getproperty, setproperty!

dictify(p) = Dict((k, getproperty(p, k)) for k in propertynames(p))

include("file_wrangling.jl")
include("data_analysis.jl")
include("column_models.jl")

include("models/kpp_optimization.jl")

end # module
