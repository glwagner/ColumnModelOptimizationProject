
#####
##### The LESbrary (so to speak)
#####

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"
LESbrary = OrderedDict(
                   # Non-rotating
                    "kato, N²: 1e-7" => (
                       filename = LESbrary_path*"kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx512_Nz256_averages.jld2",
                       stressed = true,
                       rotating = false,
                             N² = 1e-7,
                          first = 5,
                           last = 101),

                   "kato, N²: 1e-6" => (
                       filename = LESbrary_path*"kato_phillips_Nsq1.0e-06_Qu1.0e-04_Nx512_Nz256_averages.jld2",
                       stressed = true,
                       rotating = false,
                             N² = 1e-6,
                          first = 5,
                           last = 61),

                   "kato, N²: 1e-5" => (
                       filename = LESbrary_path*"kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = false,
                             N² = 1e-5,
                          first = 11,
                           last = 121),

                    "kato, N²: 1e-4" => (
                       filename = LESbrary_path*"kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2",
                       stressed = true,
                       rotating = false,
                             N² = 1e-4,
                          first = 21,
                           last = nothing),

                    # Rotating
                    "ekman, N²: 1e-7" => (
                       filename = LESbrary_path*"stress_driven_Nsq1.0e-07_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true,
                             N² = 1e-7,
                          first = 11,
                           last = 201),

                    "ekman, N²: 1e-6" => (
                       filename = LESbrary_path*"stress_driven_Nsq1.0e-06_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true,
                             N² = 1e-6,
                          first = 11,
                           last = 75),

                    "ekman, N²: 1e-5" => (
                       filename = LESbrary_path*"stress_driven_Nsq1.0e-05_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true,
                             N² = 1e-5,
                          first = 11,
                           last = nothing),

                    "ekman, N²: 1e-4" => (
                       filename = LESbrary_path*"stress_driven_Nsq1.0e-04_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
                       stressed = true,
                       rotating = true,
                             N² = 1e-4,
                          first = 11,
                           last = 301),

                    # Convection
                    "convection, N²: 2e-6" => (
                       filename = LESbrary_path*"free_convection_Qb1.0e-07_Nsq2.0e-06_Nh256_Nz256_statistics.jld2",
                       stressed = false,
                       rotating = true,
                             N² = 2e-6,
                          first = 11,
                           last = nothing),

                    "convection, N²: 1e-5" => (
                       filename = LESbrary_path*"free_convection_Qb1.0e-07_Nsq1.0e-05_Nh256_Nz256_statistics.jld2",
                       stressed = false,
                       rotating = true,
                             N² = 1e-5,
                          first = 31,
                           last = nothing),
                   )

# https://engaging-web.mit.edu/~alir/lesbrary/4DaySuite/
FourDaySuite_path = "/Users/adelinehillier/.julia/dev/4DaySuite/"
FourDaySuite = OrderedDict(
                    "4d_free_convection" => (
                        filename = FourDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                        stressed = false,
                        rotating = true,
                           first = 13, # cut out the first 2 hours
                            last = nothing),

                    "4d_strong_wind" => (
                        filename = FourDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),

                    "4d_strong_wind_no_rotation" => (
                        filename = FourDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = false,
                           first = 13,
                            last = nothing),

                     "4d_strong_wind_weak_cooling" => (
                        filename = FourDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),

                     "4d_weak_wind_strong_cooling" => (
                        filename = FourDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),
                 )

# https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/
TwoDaySuite_path = "/Users/adelinehillier/.julia/dev/2DaySuite/"
TwoDaySuite = OrderedDict(
                   "2d_free_convection" => (
                       filename = TwoDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13, # cut out the first 2 hours
                           last = nothing),

                   "2d_strong_wind" => (
                       filename = TwoDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                   "2d_strong_wind_no_rotation" => (
                       filename = TwoDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = false,
                          first = 13,
                           last = nothing),

                    "2d_strong_wind_weak_cooling" => (
                       filename = TwoDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "2d_weak_wind_strong_cooling" => (
                       filename = TwoDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )

# https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/
SixDaySuite_path = "/Users/adelinehillier/.julia/dev/6DaySuite/"
SixDaySuite = OrderedDict(
                 "6d_free_convection" => (
                     filename = SixDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                     stressed = false,
                     rotating = true,
                        first = 13, # cut out the first 2 hours
                         last = nothing),

                 "6d_strong_wind" => (
                     filename = SixDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                 "6d_strong_wind_no_rotation" => (
                     filename = SixDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = false,
                        first = 13,
                         last = nothing),

                  "6d_strong_wind_weak_cooling" => (
                     filename = SixDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                  "6d_weak_wind_strong_cooling" => (
                     filename = SixDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),
              )

GeneralStrat_path = "/Users/adelinehillier/.julia/dev/8DayLinearStrat/"
GeneralStrat = OrderedDict(
                   "general_strat_4" => (
                       filename = GeneralStrat_path*"general_strat_4/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 37, # cut out the first 6 hours
                           last = 288), # 2 days -- mixed layer depth reaches about 75 meters

                   "general_strat_8" => (
                       filename = GeneralStrat_path*"general_strat_8/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = 648), # 4 days -- mixed layer depth reaches about 75 meters

                   "general_strat_16" => (
                       filename = GeneralStrat_path*"general_strat_16/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "general_strat_32" => (
                       filename = GeneralStrat_path*"general_strat_32/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )

#####
##### KPP functionality
#####

kpp_fields(datum) = !(datum.stressed) ? (:T,) :
                    !(datum.rotating) ? (:T, :U) :
                                        (:T, :U, :V)

# kpp_relative_weights(datum) = !(datum.stressed) ? [1.0] :
#                               !(datum.rotating) ? [1.0, 1e-4] :
#                                                   [1.0, 1e-2, 1e-4]
kpp_relative_weights(datum) = !(datum.stressed) ? [1.0] :
                              !(datum.rotating) ? [1.0, 1e-4] :
                                                  [1.0, 1e-2, 1e-4]

tke_fields(datum) = !(datum.stressed) ? (:T, :e) :
                    !(datum.rotating) ? (:T, :U, :e) :
                                        (:T, :U, :V, :e)

# tke_relative_weights(datum) = !(datum.stressed) ? [1.0, 0.1] :
#                               !(datum.rotating) ? [1.0, 0.5, 0.1] :
#                                                   [1.0, 0.5, 0.5, 0.1]
tke_relative_weights(datum) = !(datum.stressed) ? [1.0, 1e-4] :
                              !(datum.rotating) ? [1.0, 1e-2, 1e-4] :
                                                  [1.0, 1e-2, 1e-2, 1e-4]
# tke_relative_weights(datum) = !(datum.stressed) ? [1.0, 1.0] :
#                               !(datum.rotating) ? [1.0, 1.0, 1.0] :
#                                                   [1.0, 1.0, 1.0, 1.0]

"Initialize a calibration run for KPP."
function init_kpp_calibration(datapath;
                                            Δz = 4,
                                            Δt = 1second,
                                  first_target = 5,
                                   last_target = nothing,
                                        fields = (:T, :U),
                              relative_weights = [1.0 for f in fields],
                              profile_analysis = ValueProfileAnalysis(),
                              # KPP-specific kwargs:
                                   mixingdepth = ModularKPP.LMDMixingDepth(),
                                      kprofile = ModularKPP.StandardCubicPolynomial(),
                                 unused_kwargs...
                              )

    data = ColumnData(datapath)
    model = ModularKPPOptimization.ColumnModel(data, Δt, Δ=Δz, mixingdepth=mixingdepth, kprofile=kprofile)

    return init_negative_log_likelihood(model, data, first_target, last_target,
                                        fields, relative_weights)
end


#####
##### TKEMassFlux functionality
#####

"Initialize a calibration run for the TKEMassFlux parameterization."
function init_tke_calibration(datapath;
                                               N = 32,
                                              Δt = 1minute,
                                    first_target = 5,
                                     last_target = nothing,
                                          fields = (:T, :U, :e),
                                relative_weights = [1.0 for f in fields],
                                #      time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage())
                                # profile_analysis = ValueProfileAnalysis(model.grid),
                                # TKE-specific kwargs:
                                   mixing_length = TKEMassFlux.SimpleMixingLength(),
                                  tke_wall_model = TKEMassFlux.PrescribedSurfaceTKEFlux(),
                              eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                                    tke_equation = TKEMassFlux.TKEParameters(),
                           convective_adjustment = nothing, # or TKEMassFlux.VariablePrandtlConvectiveAdjustment()
                                   unused_kwargs...
                              )

    data = ColumnData(datapath)

    model = TKEMassFluxOptimization.ColumnModel(data, Δt,
                                                          N = N,
                                               mixing_length = mixing_length,
                                              tke_wall_model = tke_wall_model,
                                          eddy_diffusivities = eddy_diffusivities,
                                                tke_equation = tke_equation,
                                       convective_adjustment = convective_adjustment,
                                         )

    return init_negative_log_likelihood(model, data, first_target, last_target,
                                        fields, relative_weights)
end

#####
##### Some utils common to KPP and TKEMassFlux
#####

function init_negative_log_likelihood(model, data, first_target, last_target,
                                      fields, relative_weights)

    profile_analysis = ValueProfileAnalysis(model.grid)
    # Create loss function and negative-log-likelihood object
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target
    profile_analysis = on_grid(profile_analysis, model.grid)
    weights = estimate_weights(profile_analysis, data, fields, targets, relative_weights)

    # Create loss function and NegativeLogLikelihood
    loss = LossFunction(model, data, fields=fields, targets=targets, weights=weights,
                        time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                        profile = profile_analysis)

    nll = NegativeLogLikelihood(model, data, loss)

    return nll
end
