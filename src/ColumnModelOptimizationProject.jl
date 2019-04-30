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
    summarize_data

using
    OceanTurb,
    JLD2,
    PyCall,
    Printf,
    PyPlot

dictify(p) = Dict((k, getproperty(params, k)) for k in propertynames(p))

include("file_wrangling.jl")
include("data_analysis.jl")

include("models/kpp_optimization.jl")

end # module
