using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

calibrated = (
              #=
              "tke-surface-value-mega-batch.jld2",
              "tke_calibration_surface_tke_value_ekman_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_surface_tke_value_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_surface_tke_value_kato_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_surface_tke_value_kato_N²1e-6_dz2_dt1.jld2",
              =#

              "tke-surface-value-rotating-mini-batch.jld2",

              #=
              "tke-surface-value-non-rotating-mini-batch.jld2",
              "tke-surface-value-rotating-strong-stratification-mini-batch.jld2",
              "tke_calibration_surface_tke_value_ekman_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_surface_tke_value_kato_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_surface_tke_value_kato_N²1e-5_dz2_dt1.jld2",
              =#

              #=
              "tke-surface-value-rotating-weak-stratification-mini-batch.jld2",
              "tke-surface-value-non-rotating-strong-stratification-mini-batch.jld2",
              "tke-surface-value-non-rotating-weak-stratification-mini-batch.jld2",
              "tke_calibration_surface_tke_value_ekman_N²1e-6_dz2_dt1.jld2",
              =#

              #=
              "tke-mega-batch.jld2",
              "tke_calibration_ekman_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_ekman_N²1e-5_dz2_dt1.jld2",
              "tke_calibration_kato_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_kato_N²1e-6_dz2_dt1.jld2",
              =#

              #=
              "tke-non-rotating-mini-batch.jld2",
              "tke-rotating-strong-stratification-mini-batch.jld2",
              "tke_calibration_ekman_N²1e-7_dz2_dt1.jld2",
              "tke_calibration_kato_N²1e-4_dz2_dt1.jld2",
              "tke_calibration_kato_N²1e-5_dz2_dt1.jld2",
              =#

              #=
              "tke-rotating-weak-stratification-mini-batch.jld2",
              "tke-non-rotating-strong-stratification-mini-batch.jld2",
              "tke-non-rotating-weak-stratification-mini-batch.jld2",
              "tke_calibration_ekman_N²1e-6_dz2_dt1.jld2",
              =#
             )

chunks = 10000
nchunks = 10

for case in calibrated

    casepath = joinpath("tke-data", case)
    calibration = load(casepath, "tke_calibration")

    chain = calibration.markov_chains[end]
    
    for i = 1:nchunks
        extend!(chain, chunks)
        status(chain)
        simple_safe_save(casepath, calibration)
    end
end
