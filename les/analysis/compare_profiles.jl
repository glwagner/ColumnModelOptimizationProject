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

params = DefaultFreeParameters(model, WindMixingParameters)
@show params
params = WindMixingParameters(0.25, 0.17, 0.3)

nll = NegativeLogLikelihood(model, data, relative_fields_loss)
first_link = MarkovLink(nll, params)
@show first_link.error

fig, axs = visualize_realization(params, model, data)

gcf()
