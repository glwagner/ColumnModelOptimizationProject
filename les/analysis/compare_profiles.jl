using JLD2, PyPlot, OceanTurb,
        ColumnModelOptimizationProject, Printf


using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
name = "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128"
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=5, targets=(10, 90))

model = ModularKPPOptimization.ColumnModel(data, 10minute; Δ=1.0,
        mixingdepth = ModularKPP.LMDMixingDepth()
        )

Uvariance = maxvariance(data, :U)
Vvariance = maxvariance(data, :V)
Tvariance = maxvariance(data, :T)

defaultparams = DefaultFreeParameters(model, WindMixingParameters)
params = WindMixingParameters(4.93207, 3.56363, 4.796)

#nll = NegativeLogLikelihood(model, data, relative_fields_loss)

nll = NegativeLogLikelihood(model, data, weighted_fields_loss,
    weights=(0.1 / Uvariance, 0.1 / Vvariance, 1.0 / Tvariance))

link = MarkovLink(nll, params)
default_link = MarkovLink(nll, defaultparams)

@show nll.scale = default_link.error * 1e-2

link = MarkovLink(nll, params)
default_link = MarkovLink(nll, defaultparams)

@show link.error
@show default_link.error

fig, axs = visualize_realization(params, model, data)
gcf()

fig, axs = visualize_realization(defaultparams, model, data)
gcf()
