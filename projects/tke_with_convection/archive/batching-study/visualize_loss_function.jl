using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
rc("axes", labelsize=10, titlesize=10)
rc("xtick", labelsize=10)
rc("ytick", labelsize=10)

fontsize = 8

get_position(ax) = [b for b in ax.get_position().bounds]

     datakwargs = Dict(:linewidth=>3, :alpha=>0.4, :linestyle=>"-", :color=>"k")
tke_modelkwargs = Dict(:linewidth=>2, :alpha=>0.6, :linestyle=>"--", :color=>defaultcolors[1])
        kpp_modelkwargs = Dict(:linewidth=>2, :alpha=>0.6, :linestyle=>"--", :color=>"xkcd:tomato")
kpp_default_modelkwargs = Dict(:linewidth=>1, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[2])

tke_calibration = load("tke-data/tke-mega-batch.jld2", "tke_calibration")
kpp_calibration = load("kpp-data/kpp-mega-batch.jld2", "kpp_calibration")

# Optimal parameters
tke_chain = tke_calibration.markov_chains[end]
kpp_chain = kpp_calibration.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
kpp_defaults = kpp_calibration.markov_chains[1][1].param

close("all")
fig, axs = subplots(figsize=(5.5, 1.8))

lw = 2
α = 0.6

j = 1
i = 4

tke_nll = tke_calibration.negative_log_likelihood.batch[i]
kpp_nll = kpp_calibration.negative_log_likelihood.batch[i]

data = tke_nll.data # data *should* be the same for tke and kpp

tke_model = tke_nll.model
tke_loss = tke_nll.loss

kpp_model = kpp_nll.model
kpp_default_model = deepcopy(kpp_nll.model)
kpp_loss = kpp_nll.loss

evaluate!(tke_loss, tke_c★, tke_model, data)
evaluate!(kpp_loss, kpp_c★, kpp_model, data)

f = tke_nll.model.constants.f
N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

tke_lbl = "TKE-based model"
kpp_lbl = "KPP (calibrated)"
kpp_default_lbl = "KPP (default)"

plot(tke_loss.time_series.time / hour, tke_loss.time_series.data, "-", linewidth=lw, alpha=α, label=tke_lbl,
     color=defaultcolors[1])
plot(kpp_loss.time_series.time / hour, kpp_loss.time_series.data, "--", linewidth=lw, alpha=α, label=kpp_lbl,
     color="xkcd:tomato")
     
evaluate!(kpp_loss, kpp_defaults, kpp_model, data)
plot(kpp_loss.time_series.time / hour, kpp_loss.time_series.data, ":", linewidth=lw, alpha=α, label=kpp_default_lbl,
     color=defaultcolors[1])

removespines("top", "right")
xlabel(L"t \, \mathrm{(hours)}", labelpad=-2.0)
ylabel(L"\Sigma_{\phi} \frac{\upsilon_\phi}{L_z} \int_{-L_z}^0 \left ( \Phi - \bar \phi \right )^2 \mathrm{d} z")

#=
xshift = 0.03
yshift = 0.085
pos = get_position(axs)
pos[1] += xshift
pos[2] += yshift
axs.set_position(pos)
=#

ylim(0, 0.1)
xlim(0, 26)
tight_layout()

savefig("loss-function-time-series-viz.png", dpi=480)

#
# Visualize loss function vertical profile
# 

ji = tke_loss.targets[1]
jf = tke_loss.targets[end]

initialize_and_run_until!(tke_model, data, tke_c★, ji, jf)
initialize_and_run_until!(kpp_model, data, kpp_c★, ji, jf)
initialize_and_run_until!(kpp_default_model, data, kpp_defaults, ji, jf)

grid = tke_model.grid
Δtke = CellField(grid)
Δkpp = CellField(grid)
Δkpp_default = CellField(grid)

set!(Δtke, data.T[jf])
set!(Δkpp, data.T[jf])
set!(Δkpp_default, data.T[jf])

for i in eachindex(Δtke)
    @inbounds Δtke[i] = (Δtke[i] - tke_model.solution.T[i])^2
    @inbounds Δkpp[i] = (Δkpp[i] - kpp_model.solution.T[i])^2
    @inbounds Δkpp_default[i] = (Δkpp_default[i] - kpp_default_model.solution.T[i])^2
end

fig, axs = subplots(figsize=(2.5, 2.25))

plot(Δtke, "-", color=defaultcolors[1], linewidth=2, label=tke_lbl, alpha=α)
plot(Δkpp, "--", color="xkcd:tomato", linewidth=2, label=kpp_lbl, alpha=α)
plot(Δkpp_default, ":", linewidth=2, color=defaultcolors[2], label=kpp_default_lbl, alpha=α)

removespines("top", "left")
xlabel(L"\left ( T - \bar \theta \right )^2 \, \mathrm{({}^\circ \, C)}")
ylabel(L"z \, \mathrm{(m)}")

axs.yaxis.set_label_position("right")
axs.tick_params(left=false, labelleft=false, right=true, labelright=true)

legend(prop=Dict(:size=>8), loc=3, bbox_to_anchor=(0.1, 0.0, 1.0, 1.0))

tight_layout()

#=
xshift = -0.2
yshift = -0.2
pos = get_position(axs)
pos[1] += xshift
pos[2] += yshift
axs.set_position(pos)
=#

savefig("loss-function-profile-viz.png", dpi=480)
