using
    PyPlot, PyCall, Printf,
    Dao, JLD2, Statistics,
    OceanTurb, OffsetArrays, LinearAlgebra

@use_pyplot_utils

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

font_manager = pyimport("matplotlib.font_manager")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axs = subplots(nrows=3)

sca(axs[1])
xlabel(L"C^\mathrm{Ri}")

sca(axs[2])
xlabel(L"C^\tau")

sca(axs[3])
xlabel(L"C^\mathrm{SL}")

for ax in axs
    sca(ax)
    OceanTurbPyPlotUtils.removespines("top", "right", "left")
    ax.tick_params(left=false, labelleft=false)
end

chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc"
chainname = "mcmc_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-03_std1.0e-02_032.jld2"

alpha = 0.2
bins = 200

#for name in chainnames
name = chainname

    chainpath = joinpath(chaindir, name)
    @load chainpath chain

    opt = optimal(chain)
    @show name
    @show length(chain)
    @show chain.acceptance
    @show opt.param

    samples = Dao.params(chain)
    CRi = map(x->x.CRi, samples)
    Cτ = map(x->x.Cτ, samples)
    CSL = map(x->x.CSL, samples)

    sca(axs[1])
    plt.hist(CRi, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CRi, 0, "s")

    sca(axs[2])
    plt.hist(Cτ, bins=bins, alpha=alpha, density=true)
    plot(opt.param.Cτ, 0, "s")

    sca(axs[3])
    plt.hist(CSL, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CSL, 0, "s")

#end

tight_layout()
gcf()

#
# Joint pdfs
#

fig, axs = subplots(ncols=2, nrows=2)

sca(axs[1, 1])
plt.hist2d(CRi, CSL, bins=bins)
plot(opt.param.CRi, opt.param.CSL, "r*", markersize=5)
xlabel(L"C^\mathrm{Ri}")
ylabel(L"C^\mathrm{SL}")

sca(axs[2, 1])
plt.hist2d(CRi, Cτ, bins=bins)
plot(opt.param.CRi, opt.param.Cτ, "r*", markersize=5)
xlabel(L"C^\mathrm{Ri}")
ylabel(L"C^\tau")

sca(axs[1, 2])
plt.hist2d(CSL, Cτ, bins=bins)
plot(opt.param.CSL, opt.param.Cτ, "r*", markersize=5)
xlabel(L"C^\mathrm{SL}")
ylabel(L"C^\tau")

axs[2, 2].axis("off")

tight_layout()
gcf()
