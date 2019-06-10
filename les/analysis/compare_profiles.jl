using JLD2, PyPlot, OceanTurb,
        ColumnModelOptimizationProject, Printf


using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

datadir = "data"
#filename = "simple_flux_Fb1e-09_Fu-1e-04_Nsq1e-05_Lz64_Nz256_profiles.jld2"
name = "simple_flux_Fb5e-09_Fu-1e-04_Nsq1e-06_Lz128_Nz256"
#name = "simple_flux_Fb1e-08_Fu0e+00_Nsq1e-06_Lz128_Nz256"
filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")

iters = iterations(filepath)
data = ColumnData(filepath, reversed=true, initial=1, targets=(150, 250, 350))

model = ModularKPPOptimization.ColumnModel(data, 10minute; N=128,
        #mixingdepth = ModularKPP.ROMSMixingDepth()
        mixingdepth = ModularKPP.LMDMixingDepth()
        )

defaults = DefaultFreeParameters(model, BasicParameters)

fig, axs = visualize_realization(defaults, model, data)
#fig, axs = visualize_targets(data)

gcf()
