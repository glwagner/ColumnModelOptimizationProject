using JLD2, PyPlot, Dao,
        ColumnModelOptimizationProject, Printf,
        OceanTurb, OffsetArrays, LinearAlgebra

using ColumnModelOptimizationProject.ModularKPPOptimization

@use_pyplot_utils
usecmbright()

fig, axs = subplots(figsize=(4, 4))
datadir = "data"
names = (
        "simple_flux_Fb0e+00_Fu-1e-04_Nsq1e-05_Lz64_Nz128",
        "simple_flux_Fb0e+00_Fu-1e-04_Nsq5e-06_Lz64_Nz128",
        "simple_flux_Fb0e+00_Fu-1e-04_Nsq2e-06_Lz64_Nz128",
        )

for name in names
        filepath = joinpath(@__DIR__, "..", datadir, name * "_profiles.jld2")
        data = ColumnData(filepath, reversed=true, initial=1, targets=[9, 145])
        idata = 97
        T = data.T[idata]
        #T.data .= 20 .+ T.data .- T.data[end]
        plot(T, "-", linewidth=2, alpha=0.6)
end

OceanTurbPyPlotUtils.removespines("top", "right")
xlabel("Temperature \$ ({}^\\circ \\mathrm{C})\$")
ylabel(L"z \, (\mathrm{m})")
tight_layout()
gcf()

savefig("/Users/gregorywagner/Desktop/varying_stratification_profiles.png", dpi=480)
