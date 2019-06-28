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
    "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2",
    "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2",
    "mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2",
    #"mcmc_strat_simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-06_Lz64_Nz128_s3.0e-04_dt5.0_Δ2.0.jld2"
    )

Δ, N² = [], []

markers = [
    "*",
    "^",
    "s",
    "p"]

markerstyle = Dict(
     :linestyle => "None",
    :markersize => 8,
    )

chainpath = joinpath(chaindir, chainnames[1])
@load chainpath chain
C₀ = chain[1].param
paramnames = propertynames(C₀)
nparams = length(paramnames)

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axs = subplots(nrows=nparams, figsize=(9, 5.5))

for (i, ax) in enumerate(axs)
    p = paramnames[i]
    sca(ax)
    xlabel(latexparams[p])
    OceanTurbPyPlotUtils.removespines("top", "right", "left")
    ax.tick_params(left=false, labelleft=false)
end

ρCmax = zeros(nparams)
C★ = [zeros(nparams) for name in chainnames]

for (i, name) in enumerate(chainnames)
    global C₀

    c = defaultcolors[i]

    chainpath = joinpath(chaindir, name)
    @show chainpath
    @load chainpath chain
    C₀ = chain[1].param

    opt = optimal(chain)
    @show name
    @show length(chain)
    @show chain.acceptance
    @show opt.param

    push!(Δ, chain.nll.model.grid.Δc)
    push!(N², chain.nll.data.bottom_Bz)

    after = 1000
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

    for (j, Cname) in enumerate(paramnames)

        lbl₀ = "Large et al. (1994)"
        lbl★ = @sprintf(
            "\$ N^2 = %.0e \\, \\mathrm{s^{-2}}\$, \$ \\Delta = %.0f \$ m", N²[i], Δ[i])

        sca(axs[j])
        i == 1 && plot(C₀[j], 1.2ρCmax[j], label=lbl₀, marker="o", color="0.1",
                        linestyle="None", markersize=8, alpha=0.8)

        marker = markers[i]
        plot(C★[i][j], 1.2ρCmax[j], marker; label=lbl★, color=c, markerstyle...)

    end
end

legendkw = Dict(
    :markerscale=>1.2, :fontsize=>10,
    :loc=>"upper left", :bbox_to_anchor=>(1.2, 1.0),
    :frameon=>true, :framealpha=>1.0)

sca(axs[1])
xlim(0.2, 0.7)

sca(axs[2])
legend(; legendkw...)
#xlim(0.0, 0.4)

sca(axs[3])
xlim(0.2, 0.42)

tight_layout()
gcf()

#savefig("/Users/gregorywagner/Desktop/pdf_examples.png", dpi=480)
savefig("/Users/gregorywagner/Desktop/stratification_variation.png", dpi=480)
