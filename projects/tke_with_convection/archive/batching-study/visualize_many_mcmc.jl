using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

calibrated = (
              # Prescribed surface value runs
              "tke-scaled-flux-mega-batch.jld2",

              "tke-scaled-flux-rotating-mini-batch-continuation.jld2", 
              "tke-scaled-flux-non-rotating-mini-batch-continuation.jld2",
              "tke-scaled-flux-rotating-weak-stratification-mini-batch-continuation.jld2", 
              "tke-scaled-flux-non-rotating-weak-stratification-mini-batch-continuation.jld2",
              "tke-scaled-flux-rotating-strong-stratification-mini-batch-continuation.jld2", 
              "tke-scaled-flux-non-rotating-strong-stratification-mini-batch-continuation.jld2",

              "tke_calibration_scaled_flux_ekman_N²1e-4_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-5_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-6_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-7_dz2_dt1-continuation.jld2",

              "tke_calibration_scaled_flux_kato_N²1e-4_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-5_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-6_dz2_dt1-continuation.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-7_dz2_dt1-continuation.jld2",

              #=
              "tke-scaled-flux-mega-batch.jld2",
              "tke-scaled-flux-non-rotating-mini-batch.jld2",

              "tke-scaled-flux-rotating-strong-stratification-mini-batch.jld2",
              "tke-scaled-flux-rotating-weak-stratification-mini-batch.jld2",
              "tke-scaled-flux-non-rotating-strong-stratification-mini-batch.jld2",
              "tke-scaled-flux-non-rotating-weak-stratification-mini-batch.jld2",

              "tke_calibration_scaled_flux_ekman_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_ekman_N²1e-6_dz2_dt1.jld2",

              "tke_calibration_scaled_flux_kato_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-6_dz2_dt1.jld2",
              "tke_calibration_scaled_flux_kato_N²1e-7_dz2_dt1.jld2",
              =#

              #=
              # Prescribed surface flux runs
              "tke-mega-batch.jld2",

              "tke-non-rotating-mini-batch.jld2",

              "tke-rotating-weak-stratification-mini-batch.jld2",
              "tke-rotating-strong-stratification-mini-batch.jld2",
              "tke-non-rotating-strong-stratification-mini-batch.jld2",
              "tke-non-rotating-weak-stratification-mini-batch.jld2",

              "tke_calibration_simple_ekman_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_simple_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_simple_ekman_N²1e-6_dz2_dt1.jld2",
              "tke_calibration_simple_ekman_N²1e-7_dz2_dt1.jld2",

              "tke_calibration_simple_kato_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-6_dz2_dt1.jld2",
              "tke_calibration_simple_kato_N²1e-7_dz2_dt1.jld2",
              =#
             )

for case in calibrated
    casepath = joinpath("tke-data", case)
    calibration = load(casepath, "calibration")

    chain = calibration.markov_chains[end]
    
    fig, axs, ρ = visualize_markov_chain!(
        chain, parameter_latex_guide=TKEMassFluxOptimization.parameter_latex_guide)

    fig.suptitle(replace(case, ['_', '-'] => " "))
    tight_layout()

    pause(0.1)
end
