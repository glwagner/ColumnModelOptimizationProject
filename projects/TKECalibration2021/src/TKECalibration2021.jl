module TKECalibration2021

using Statistics, Distributions, PyPlot, OrderedCollections, JLD2, FileIO, Printf
using OceanTurb, Dao
using ColumnModelOptimizationProject
using ColumnModelOptimizationProject.TKEMassFluxOptimization
using ColumnModelOptimizationProject.TKEMassFlux: FluxProportionalConvectiveAdjustment
using Optim
using Dao: AdaptiveAlgebraicSchedule

export
       parameter_latex_guide,

       # setup.jl
       LESbrary,
       FourDaySuite,
       GeneralStrat,

       init_kpp_calibration,
       init_tke_calibration,

       # utils_writing_output.jl
       saveplot,
       open_output_file,
       writeout,

       # algorithms.jl
       simulated_annealing,
       nelder_mead,
       l_bfgs,
       random_plugin,
       gradient_descent,

       # free_parameters.jl
       TKEParametersRiDependent,
       TKEParametersRiIndependent,
       TKEParametersConvectiveAdjustmentRiDependent,
       TKEParametersConvectiveAdjustmentRiIndependent,
       KPPWindMixingParameters,
       KPPWindMixingOrConvectionParameters,
       TKEFreeConvection,
       TKEFreeConvectionConvectiveAdjustmentRiDependent,
       TKEFreeConvectionConvectiveAdjustmentRiIndependent,
       TKEFreeConvectionRiIndependent,

       # TKECalibration2021
       custom_tke_calibration

include("../../../src/ColumnModelOptimizationProject.jl")
include("setup.jl")
include("utils.jl")
include("utils_writing_output.jl")
include("free_parameters.jl")
include("algorithms.jl")

ID = OceanTurb.TKEMassFlux.IndependentDiffusivities()
DD = OceanTurb.TKEMassFlux.RiDependentDiffusivities()
CA = OceanTurb.TKEMassFlux.FluxProportionalConvectiveAdjustment()

parameter_specific_kwargs = Dict(

   TKEParametersRiDependent => (eddy_diffusivities = DD,
                                convective_adjustment = nothing,
                               ),

   TKEParametersRiIndependent => (eddy_diffusivities = ID,
                                  convective_adjustment = nothing,
                               ),

   TKEParametersConvectiveAdjustmentRiDependent => (eddy_diffusivities = DD,
                                convective_adjustment = CA,
                               ),

   TKEParametersConvectiveAdjustmentRiIndependent => (eddy_diffusivities = ID,
                                convective_adjustment = CA,
                               ),

   TKEFreeConvection => (eddy_diffusivities = DD,
                                convective_adjustment = nothing,
                               ),

   TKEFreeConvectionConvectiveAdjustmentRiDependent => (eddy_diffusivities = DD,
                                convective_adjustment = CA,
                               ),

   TKEFreeConvectionConvectiveAdjustmentRiIndependent => (eddy_diffusivities = ID,
                                convective_adjustment = CA,
                               ),

   TKEFreeConvectionRiIndependent => (eddy_diffusivities = ID,
                                convective_adjustment = nothing,
                               ),
)

# Run forward map and then compute loss from forward map output
# ℱ = model_time_series(default_parameters, model, cdata, loss_function)
# myloss(ℱ) = loss_function(ℱ, cdata)
# myloss(ℱ)

function custom_defaults(model, RelevantParameters)
    fields = fieldnames(RelevantParameters)
    defaults = DefaultFreeParameters(model, RelevantParameters)

    set_if_present!(defaults, :Cᴰ,  0.01)
    set_if_present!(defaults, :Cᴸᵇ, 0.02)
    set_if_present!(defaults, :CʷwΔ, 5.0)

    # Independent diffusivities
    set_if_present!(defaults, :Cᴷc, 0.5)
    set_if_present!(defaults, :Cᴷe, 0.02)

    set_if_present!(defaults, :CᴷRiʷ, 0.5)
    set_if_present!(defaults, :CᴷRiᶜ, -0.73)
    set_if_present!(defaults, :Cᴷu⁻, 0.7062)
    set_if_present!(defaults, :Cᴷu⁺, 0.0071)
    set_if_present!(defaults, :Cᴷc⁻, 4.5015)
    set_if_present!(defaults, :Cᴷe⁻, 1.2482)
    set_if_present!(defaults, :Cᴷe⁺, 0.0323)
    set_if_present!(defaults, :Cʷu★, 4.1025)
    set_if_present!(defaults, :CʷwΔ, 4.1025)

    return defaults
end

"""
This is an extension to `init_tke_calibration` for specific cases used in March 2021 calibration experiments.
If LESdata is just a path (String), will return the NegativeLogLikelihood object for that simulation;
If LESdata is one of the OrderedDicts from "utils.jl", will return the BatchedNegativeLogLikelihood object for those simulations.
"""
function custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize)

    function get_nll(LEScase)
        nll = init_tke_calibration(LEScase.filename;
                                                 N = 32,
                                                Δt = 60.0, #1minute
                                      first_target = LEScase.first,
                                       last_target = LEScase.last,
                                            fields = tke_fields(LEScase),
                                  relative_weights = tke_relative_weights(LEScase),
                                  parameter_specific_kwargs[RelevantParameters]...
                                )
        set!(nll.model, custom_defaults(nll.model, RelevantParameters))
        initial_parameters = custom_defaults(nll.model, ParametersToOptimize)
        return nll, initial_parameters
    end

    # Single simulation
    if typeof(LESdata) <: NamedTuple
        return get_nll(LESdata)
    end

    # Batched
    batch = []
    initial_parameters = nothing
    for LEScase in values(LESdata)
        nll, initial_parameters = get_nll(LEScase)
        push!(batch, nll)
    end

    weights = [1.0 for d in LESdata]

    batched_nll = BatchedNegativeLogLikelihood([nll for nll in batch], weights=weights)
    return batched_nll, initial_parameters
end

end # module
