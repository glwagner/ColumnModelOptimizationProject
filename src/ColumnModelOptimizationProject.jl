module ColumnModelOptimizationProject

export
    dictify,
    FreeParameters,
    DefaultFreeParameters,
    get_free_parameters,
    @free_parameters,

    # utils
    variance,
    max_variance,
    mean_variance,
    max_gradient_variance,
    initialize_forward_run!,
    initialize_and_run_until!,
    simple_safe_save,

    # file_wrangling.jl
    get_iterations,
    get_times,
    get_data,
    get_parameter,
    get_grid_params,

    # column_models.jl
    ColumnData,

    # visualization.jl
    defaultcolors,
    removespine,
    removespines,
    plot_data!,
    format_axs!,
    visualize_realizations,
    visualize_loss_function,
    visualize_markov_chain!,
    plot_loss_function,

    # loss_functions.jl
    evaluate!,
    analyze_weighted_profile_discrepancy,
    VarianceWeights,
    LossFunction,
    TimeSeriesAnalysis,
    TimeAverage,
    ValueProfileAnalysis,
    GradientProfileAnalysis,
    on_grid,

    # forward_map.jl
    ModelTimeSeries,
    model_time_series,

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
    FileIO,
    OffsetArrays,
    Printf,
    Dao

include(joinpath(pathof(OceanTurb), "..", "..", "plotting", "OceanTurbPyPlotUtils.jl") )

using PyPlot, PyCall, .OceanTurbPyPlotUtils

import OceanTurb: set!, absolute_error, run_until!

import Base: length

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

function Base.similar(p::FreeParameters{N, T}) where {N, T}
    P = typeof(p).name.wrapper
    return P((zero(T) for i=1:N)...)
end

Base.show(io::IO, p::FreeParameters) = print(io, "$(typeof(p)):", '\n',
                                             @sprintf("% 24s: ", "parameter names"),
                                             (@sprintf("%-8s", n) for n in propertynames(p))..., '\n',
                                             @sprintf("% 24s: ", "values"),
                                             (@sprintf("%-8.4f", pᵢ) for pᵢ in p)...)

dictify(p) = Dict((k, getproperty(p, k)) for k in propertynames(p))

set!(::Nothing, args...) = nothing # placeholder

function DefaultFreeParameters(cm, freeparamtype)
    paramnames, paramtypes = get_free_parameters(cm)

    alldefaults = (ptype() for ptype in values(paramtypes))

    freeparams = []
    for pname in fieldnames(freeparamtype)
        for ptype in alldefaults
            pname ∈ propertynames(ptype) && push!(freeparams, getproperty(ptype, pname))
        end
    end

    return eval(Expr(:call, freeparamtype, freeparams...))
end

macro free_parameters(GroupName, parameter_names...)
    N = length(parameter_names)
    parameter_exprs = [:($name :: T; ) for name in parameter_names]
    return esc(quote
        Base.@kwdef mutable struct $GroupName{T} <: FreeParameters{$N, T}
            $(parameter_exprs...)
        end
    end)
end

include("utils.jl")
include("file_wrangling.jl")
include("models_and_data.jl")
include("loss_functions.jl")
include("visualization.jl")
include("data_analysis.jl")
include("forward_map.jl")

include("ModularKPPOptimization/ModularKPPOptimization.jl")
include("TKEMassFluxOptimization/TKEMassFluxOptimization.jl")

end # module
