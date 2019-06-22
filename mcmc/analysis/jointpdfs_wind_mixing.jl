using
    PyPlot, PyCall, Printf,
    Dao, JLD2, Statistics,
    OceanTurb, OffsetArrays, LinearAlgebra

@use_pyplot_utils

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-03_std1.0e-02_032.jld2"
alpha = 0.2
bins = 200

font_manager = pyimport("matplotlib.font_manager")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

markerstyle = Dict(
         :color => "xkcd:tomato",
     :linestyle => nothing,
        :marker => "*",
    :markersize => 8,
    )

#
# Joint pdfs
#

ϕ = 0:0.01:2π
xc(r, x0=0) = @. x0 + r * cos(ϕ)
yc(r, y0=0) = @. y0 + r * sin(ϕ)

fig, axs = subplots(ncols=2, nrows=2)

sca(axs[1, 1])
plt.hist2d(CRi, CSL, bins=bins)
plot(opt.param.CRi, opt.param.CSL; markerstyle...)
xlabel(L"C^\mathrm{Ri}")
ylabel(L"C^\mathrm{SL}")
xlim(0.29, 0.38)
ylim(0.55, 1.0)

#r = 0.01
#plot(xc(r, 0.31), yc(r, 0.8), "r-")

sca(axs[2, 1])
plt.hist2d(CRi, Cτ, bins=bins)
plot(opt.param.CRi, opt.param.Cτ; markerstyle...)
xlabel(L"C^\mathrm{Ri}")
ylabel(L"C^\tau")
xlim(0.28, 0.37)
ylim(0.29, 0.37)
axs[2, 1].set_aspect(1)

sca(axs[1, 2])
plt.hist2d(CSL, Cτ, bins=bins)
plot(opt.param.CSL, opt.param.Cτ; markerstyle...)
xlabel(L"C^\mathrm{SL}")
ylabel(L"C^\tau")
xlim(0.5, 1.0)
ylim(0.29, 0.37)

axs[2, 2].axis("off")

tight_layout()
gcf()
