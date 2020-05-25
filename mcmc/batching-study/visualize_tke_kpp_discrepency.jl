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

function thin(kwargs)
    new_kwargs = deepcopy(kwargs)
    new_kwargs[:linewidth] = kwargs[:linewidth]/2
    return new_kwargs
end

function plot_data_field!(ax, fieldname, data, ji, jf, datakwargs)
    sca(ax)
    ϕ_data_i = getproperty(data, fieldname)[ji]
    ϕ_data_f = getproperty(data, fieldname)[jf]
    plot(ϕ_data_f; datakwargs...)
    return nothing
end

function plot_model_field!(ax, fieldname, model, modelkwargs)
    sca(ax)
    ϕ_model = getproperty(model.solution, fieldname)
    plot(ϕ_model; modelkwargs...)
    return nothing
end

tke_calibration = load("tke-data/tke-scaled-flux-mega-batch.jld2", "calibration")
kpp_calibration = load("kpp-data/kpp-mega-batch.jld2", "kpp_calibration")

data_label = "LES"
tke_label = "TKE-based model"
kpp_label = "KPP (calibrated)"
kpp_default_label = "KPP (default)"

# Optimal parameters
tke_chain = tke_calibration.markov_chains[end]
kpp_chain = kpp_calibration.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
@show kpp_defaults = kpp_calibration.markov_chains[1][1].param

ncases = length(tke_calibration.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=3, nrows=1, figsize=(6.5, 2.8))

N²s = []
fs = []

for (j, i) = enumerate([2, 3, 4])
    tke_nll = tke_calibration.negative_log_likelihood.batch[i]
    kpp_nll = kpp_calibration.negative_log_likelihood.batch[i]

    f = tke_nll.model.constants.f
    N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

    push!(fs, f)
    push!(N²s, N²)

    data = tke_nll.data # data *should* be the same for tke and kpp

    tke_model = tke_nll.model
    tke_loss = tke_nll.loss

    kpp_model = kpp_nll.model
    kpp_default_model = deepcopy(kpp_nll.model)
    kpp_loss = kpp_nll.loss

    ji = tke_loss.targets[1]
    jf = tke_loss.targets[end]

    initialize_and_run_until!(tke_model, data, tke_c★, ji, jf)
    initialize_and_run_until!(kpp_model, data, kpp_c★, ji, jf)
    initialize_and_run_until!(kpp_default_model, data, kpp_defaults, ji, jf)

    ax = axs[j]
     plot_data_field!(ax, :T, data, ji, jf, merge(datakwargs, Dict(:label=>data_label)))
    plot_model_field!(ax, :T, tke_model, merge(tke_modelkwargs, Dict(:label=>tke_label)))
    plot_model_field!(ax, :T, kpp_model, merge(kpp_modelkwargs, Dict(:label=>kpp_label)))
    plot_model_field!(ax, :T, kpp_default_model, merge(kpp_default_modelkwargs, Dict(:label=>kpp_default_label)))

    ax.ticklabel_format(useOffset=false)
end

ax = axs[1]
sca(ax)
xlabel(L"T \, \mathrm{({}^\circ \, C)}")
ylabel(L"z \, \mathrm{(m)}")
removespines("top", "right") #, "left", "bottom")

text(0.11, 0.84, @sprintf(" \$ N^2 = 10^{%d} \\, \\mathrm{s^{-2}} \$", log10(N²s[1])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom")

text(0.16, 0.80, @sprintf(" \$ f = 10^{%d} \\, \\mathrm{s^{-1}} \$", log10(fs[1])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="top")

ax = axs[2]
sca(ax)
xlabel(L"T \, \mathrm{({}^\circ \, C)}")
axs[2].tick_params(left=false, labelleft=false)
removespines("top", "right", "left") #, "left", "bottom")

text(0.45, 0.18, @sprintf(" \$ N^2 = 10^{%d} \\, \\mathrm{s^{-2}} \$", log10(N²s[2])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom")

text(0.50, 0.14, @sprintf(" \$ f = 10^{%d} \\, \\mathrm{s^{-1}} \$", log10(fs[2])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="top")

ax = axs[3]
sca(ax)
xlabel(L"T \, \mathrm{({}^\circ \, C)}")
axs[3].tick_params(left=false, labelleft=false)
removespines("top", "right", "left") #, "left", "bottom")

text(0.45, 0.18, @sprintf(" \$ N^2 = 10^{%d} \\, \\mathrm{s^{-2}} \$", log10(N²s[3])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom")

text(0.50, 0.14, @sprintf(" \$ f = 10^{%d} \\, \\mathrm{s^{-1}} \$", log10(fs[3])),
     transform=ax.transAxes, horizontalalignment="left", verticalalignment="top")

xshift = 0.03
yshift = 0.05
for ax in axs
    pos = get_position(ax)
    pos[1] += xshift
    pos[2] += yshift
    ax.set_position(pos)
end

sca(axs[2])
leg = legend(markerfirst=true, loc=3, bbox_to_anchor=(-0.13, 0.6, 1.0, 0.5), 
         prop=Dict(:size=>8))

savefig("kpp_versus_tke.png", dpi=480)
