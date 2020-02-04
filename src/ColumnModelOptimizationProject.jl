module ColumnModelOptimizationProject

export
    dictify,
    FreeParameters,
    DefaultFreeParameters,
    get_free_parameters,

    # utils
    variance,
    max_variance,
    initialize_and_run_until!,

    # file_wrangling.jl
    get_iterations,
    get_times,
    get_data,
    get_parameter,
    get_grid_params,

    # column_models.jl
    ColumnData,

    # visualization.jl
    visualize_realizations,
    plot_loss_function,

    # loss_functions.jl
    evaluate!,
    weighted_error,
    VarianceWeights,
    TimeAveragedLossFunction,

    # data_analysis.jl
    removespines,
    summarize_data,

    # models/
    ModularKPPOptimization,
    TKEMassFluxOptimization

using
    Statistics,
    StaticArrays,
    OceanTurb,
    JLD2,
    OffsetArrays,
    Printf

include(joinpath(pathof(OceanTurb), "..", "..", "plotting", "OceanTurbPyPlotUtils.jl") )

using PyPlot, PyCall, .OceanTurbPyPlotUtils

import OceanTurb: set!, absolute_error

import Base: length

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

function DefaultFreeParameters(cm, freeparamtype)
    paramnames, paramtypes = get_free_parameters(cm)

    alldefaults = (ptype() for ptype in values(paramtypes))

    freeparams = []
    for pname in fieldnames(freeparamtype)
        for ptype in alldefaults
            pname ∈ propertynames(ptype) && push!(freeparams, getproperty(ptype, pname))
        end
    end

    eval(Expr(:call, freeparamtype, freeparams...))
end

include("utils.jl")
include("file_wrangling.jl")
include("models_and_data.jl")
include("loss_functions.jl")
include("visualization.jl")
include("data_analysis.jl")

include("ModularKPPOptimization/ModularKPPOptimization.jl")
include("TKEMassFluxOptimization/TKEMassFluxOptimization.jl")

end # module
