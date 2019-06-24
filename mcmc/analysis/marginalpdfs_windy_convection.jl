using
    PyPlot, PyCall, Printf,
    Dao, JLD2, Statistics,
    OceanTurb, OffsetArrays, LinearAlgebra

@use_pyplot_utils
OceanTurbPyPlotUtils.usecmbright()

using
    ColumnModelOptimizationProject,
    ColumnModelOptimizationProject.ModularKPPOptimization

alpha = 0.2
bins = 200
chaindir = "/Users/gregorywagner/Projects/ColumnModelOptimizationProject.jl/mcmc/data"
chainnames = (
    "mcmc_simple_flux_Fb1e-08_Fu-1e-04_Nsq2e-06_Lz64_Nz256_e1.0e-03_2.jld2",
    )

paramnames = (:CRi, :CSL, :Cτ)
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

markerstyle = Dict(
     :linestyle => "None",
    :markersize => 6,
    )

fig, axs = subplots(nrows=3)

sca(axs[1])
xlabel(L"C^\mathrm{Ri}")

sca(axs[2])
xlabel(L"C^\mathrm{SL}")

sca(axs[3])
xlabel(L"C^\tau")

for ax in axs
    sca(ax)
    OceanTurbPyPlotUtils.removespines("top", "right", "left")
    ax.tick_params(left=false, labelleft=false)
end

ρCmax = zeros(3)
C★ = [zeros(3) for name in chainnames]

for (i, name) in enumerate(chainnames)
    global C₀

    c = defaultcolors[i]

    chainpath = joinpath(chaindir, name)
    @load chainpath chain
    C₀ = DefaultFreeParameters(chain.nll.model, WindMixingParameters)

    opt = optimal(chain)
    @show name
    @show length(chain)
    @show chain.acceptance
    @show opt.param

    after = 1
    samples = Dao.params(chain, after=after)

    for (j, Cname) in enumerate(paramnames)

        C = map(x->getproperty(x, Cname), samples)
        C★[i][j] = getproperty(opt.param, Cname)

        sca(axs[j])
        ρCj, _, _ = plt.hist(C, bins=bins, alpha=alpha, density=true, facecolor=c)

        ρCmax[j] = max(ρCmax[j], maximum(ρCj))
    end
end


for (i, name) in enumerate(chainnames)
    c = defaultcolors[i]

    for (j, Cname) in enumerate((:CRi, :Cτ, :CSL))

        lbl★ = "" #j == 2 ? @sprintf("\$ \\Delta = %.0f \$ m", Δ[i]) : ""
        lbl₀ = "" #j == 2 ? "Large et al. (1994)" : ""

        sca(axs[j])
        i == 1 && plot(C₀[j], 1.1ρCmax[j], label=lbl₀, marker="o", color="0.1",
                        linestyle="None", markersize=4, alpha=0.8)
        plot(C★[i][j], 1.1ρCmax[j], "*"; label=lbl★, color=c, markerstyle...)

    end
end

legendkw = Dict(
    :markerscale=>1.2, :fontsize=>10,
    :loc=>"center", :bbox_to_anchor=>(0.4, 0.7),
    :frameon=>true, :framealpha=>0.5)

sca(axs[2])
legend(; legendkw...)

tight_layout()
gcf()
