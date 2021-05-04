module TKECalibration2021

using Statistics, Distributions, PyPlot, OrderedCollections, JLD2, FileIO, Printf
using Dao, OceanTurb
using ColumnModelOptimizationProject
using ColumnModelOptimizationProject.TKEMassFluxOptimization
using ColumnModelOptimizationProject.TKEMassFlux: VariablePrandtlConvectiveAdjustment
using Optim
using Dao: AdaptiveAlgebraicSchedule
using Statistics, Distributions
using OrderedCollections

using LinearAlgebra
using Random
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

export
       parameter_latex_guide,

       # setup.jl
       LESbrary,
       TwoDaySuite,
       FourDaySuite,
       SixDaySuite,
       GeneralStrat,

       init_kpp_calibration,
       init_tke_calibration,

       get_nll,
       Parameters,
       DataSet,
       dataset,
       CalibrationExperiment,
       # validation_loss_reduction,

       # visuals.jl
       visualize_and_save,

       # utils_writing_output.jl
       saveplot,
       open_output_file,
       writeout,

       # algorithms.jl
       ensemble_kalman_inversion,
       get_bounds_and_variance,
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
       custom_tke_calibration,

       # ColumnModelOptimizationProject
       visualize_realizations,
       FreeParameters,
       @free_parameters,
       set!

include("../../../src/ColumnModelOptimizationProject.jl")
include("setup.jl")
include("utils.jl")
include("utils_writing_output.jl")
include("free_parameters.jl")
include("algorithms.jl")
include("visuals.jl")

ID = OceanTurb.TKEMassFlux.IndependentDiffusivities()
DD = OceanTurb.TKEMassFlux.RiDependentDiffusivities()
CA = OceanTurb.TKEMassFlux.VariablePrandtlConvectiveAdjustment()

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

    set_if_present!(defaults, :Cᴰ,  0.5915)
    set_if_present!(defaults, :Cᴸᵇ, 1.1745)
    set_if_present!(defaults, :CʷwΔ, 5.0)

    # Independent diffusivities
    set_if_present!(defaults, :Cᴷc, 0.09518)
    set_if_present!(defaults, :Cᴷe, 0.055)
    set_if_present!(defaults, :Cᴷu, 0.1243)

    # Convective Adjustment
    set_if_present!(defaults, :Cᴬc, 0.6706)
    set_if_present!(defaults, :Cᴬu, 0.0057)
    set_if_present!(defaults, :Cᴬe, 0.2717)

    set_if_present!(defaults, :CᴷRiʷ, 0.5)
    set_if_present!(defaults, :CᴷRiᶜ, -0.73)
    set_if_present!(defaults, :Cᴷu⁻, 0.7062)
    set_if_present!(defaults, :Cᴷu⁺, 0.0071)
    set_if_present!(defaults, :Cᴷc⁻, 4.5015)
    set_if_present!(defaults, :Cᴷe⁻, 1.2482)
    set_if_present!(defaults, :Cᴷe⁺, 0.0323)
    set_if_present!(defaults, :Cʷu★, 6.645)
    set_if_present!(defaults, :CʷwΔ, 0.0579)

    return defaults
end

Base.@kwdef struct Parameters{T <: UnionAll}
    RelevantParameters::T
    ParametersToOptimize::T
end

struct DataSet{T, D, NLL, F, FP}
        LESdata::T
        relative_weights::D # field weights
        nll::NLL
        nll_wrapper::F
        default_parameters::FP
end

struct CalibrationExperiment{DS, P, FP}
        calibration::DS
        validation::DS
        parameters::P
        default_parameters::FP
        # optimal_parameters::FreeParameters # where to begin calibration
end

function CalibrationExperiment(calibration, validation, parameters)
    CalibrationExperiment(calibration, validation, parameters, calibration.default_parameters)
end

function get_nll(LEScase, p::Parameters, relative_weights)

    fields = !(LEScase.stressed) ? (:T, :e) :
             !(LEScase.rotating) ? (:T, :U, :e) :
                                   (:T, :U, :V, :e)

    relative_weights_ = [relative_weights[field] for field in fields]

    nll = init_tke_calibration(LEScase.filename;
                                             N = 32,
                                            Δt = 60.0, #1minute
                                  first_target = LEScase.first,
                                   last_target = LEScase.last,
                                        fields = fields,
                              relative_weights = relative_weights_,
                              parameter_specific_kwargs[p.RelevantParameters]...
                            )

    # Set model to custom defaults
    set!(nll.model, custom_defaults(nll.model, p.RelevantParameters))
    default_parameters = custom_defaults(nll.model, p.ParametersToOptimize)

    return nll, default_parameters
end

function dataset(LESdata, p::Parameters{UnionAll}; relative_weights = Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0))

    if typeof(LESdata) <: NamedTuple

        # Single simulation
        nll, default_parameters = get_nll(LESdata, p, relative_weights)

    else

        # Batched
        batch = []
        default_parameters = nothing
        for LEScase in values(LESdata)
            nll, default_parameters = get_nll(LEScase, p, relative_weights)
            push!(batch, nll)
        end
        nll = BatchedNegativeLogLikelihood([nll for nll in batch],
                                            weights=[1.0 for d in LESdata])
    end

    # Velossiwrapper 🐉 wrapper for vector input
    nll_wrapper(θ::Vector) = nll(p.ParametersToOptimize(θ))

    return DataSet(LESdata, relative_weights, nll, nll_wrapper, default_parameters)
end

end # module
