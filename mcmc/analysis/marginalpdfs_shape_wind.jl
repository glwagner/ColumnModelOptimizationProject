using
    PyPlot, PyCall, Printf,
    Dao, JLD2, Statistics,
    OceanTurb, OffsetArrays, LinearAlgebra

@use_pyplot_utils

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
#chainname = "mcmc_shape_bigjumps_nodiff_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e5.0e-03_std1.0e-02_032.jld2"
#chainname = "mcmc_shape_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-02_064.jld2"
chainname = "mcmc_shape_smallstd_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz128_Nz512_e1.0e-03_128.jld2"
#chainname = "mcmc_shape_smallstd_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_e1.0e-01_064.jld2"

font_manager = pyimport("matplotlib.font_manager")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axs = subplots(nrows=5, figsize=(6, 6))

sca(axs[1])
xlabel(L"C^\mathrm{Ri}")

sca(axs[2])
xlabel(L"C^\tau")

sca(axs[3])
xlabel(L"C^\mathrm{SL}")

sca(axs[4])
xlabel(L"C^{S_0}")

sca(axs[5])
xlabel(L"C^{S_1}")

for ax in axs
    sca(ax)
    OceanTurbPyPlotUtils.removespines("top", "right", "left")
    ax.tick_params(left=false, labelleft=false)
end


alpha = 0.2
bins = 100

#for name in chainnames
name = chainname

    chainpath = joinpath(chaindir, name)
    @load chainpath chain

    opt = optimal(chain)
    @show name
    @show chain.acceptance
    @show opt.param

    samples = Dao.params(chain)
    CRi = map(x->x.CRi, samples)
    Cτ = map(x->x.Cτ, samples)
    CSL = map(x->x.CSL, samples)
    CS0 = map(x->x.CS0, samples)
    CS1 = map(x->x.CS1, samples)
    sca(axs[1])
    plt.hist(CRi, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CRi, 0, "s")

    sca(axs[2])
    plt.hist(Cτ, bins=bins, alpha=alpha, density=true)
    plot(opt.param.Cτ, 0, "s")

    sca(axs[3])
    plt.hist(CSL, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CSL, 0, "s")

    sca(axs[4])
    plt.hist(CS0, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CS0, 0, "s")

    sca(axs[5])
    plt.hist(CS1, bins=bins, alpha=alpha, density=true)
    plot(opt.param.CS1, 0, "s")

#end

tight_layout()
gcf()
