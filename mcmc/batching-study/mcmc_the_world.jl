
using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

calibrated = (
              "tke-mega-batch.jld2",
              "tke_calibration_simple_ekman_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_simple_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-6_dz2_dt1.jld2",

              "tke-non-rotating-mini-batch.jld2",
              "tke-rotating-strong-stratification-mini-batch.jld2",
              "tke_calibration_simple_ekman_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-5_dz2_dt1.jld2",

              "tke-rotating-weak-stratification-mini-batch.jld2",
              "tke-non-rotating-strong-stratification-mini-batch.jld2",
              "tke-non-rotating-weak-stratification-mini-batch.jld2",
              "tke_calibration_simple_ekman_N²1e-6_dz2_dt1.jld2",
             )

chunks = 100000
nchunks = 3

for case in calibrated

    calibration = try
        load(joinpath("tke-data", case), "calibration")
    catch
        load(joinpath("tke-data", case), "tke_calibration")
    end

    chain = calibration.markov_chains[end]
    
    fig, axs, ρ = visualize_markov_chain!(
        chain, parameter_latex_guide=TKEMassFluxOptimization.parameter_latex_guide)

    fig.suptitle(replace(case, ['_', '-'] => " "))
    tight_layout()

    pause(0.1)

    #=
    
    for i = 1:nchunks
        extend!(chain, chunks)
        simple_safe_save(case, calibration)
    end
    
    =#
end
