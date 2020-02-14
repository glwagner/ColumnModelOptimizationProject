using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

fontsize = 8
fs = 12

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")

 default_datakwargs = Dict(:linewidth=>3, :alpha=>0.35, :linestyle=>"-", :color=>"k")
default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.9, :linestyle=>"--", :color=>defaultcolors[1])

get_position(ax) = [b for b in ax.get_position().bounds]

function thin(kwargs)
    new_kwargs = deepcopy(kwargs)
    new_kwargs[:linewidth] = kwargs[:linewidth]/2
    return new_kwargs
end

function plot_data_field!(ax, fieldname, data, ji, jf, default_datakwargs)
    sca(ax)
    ϕ_data_i = getproperty(data, fieldname)[ji]
    ϕ_data_f = getproperty(data, fieldname)[jf]
    plot(ϕ_data_f; default_datakwargs...)
    return nothing
end

function plot_model_field!(ax, fieldname, model, default_modelkwargs)
    sca(ax)
    ϕ_model = getproperty(model.solution, fieldname)
    plot(ϕ_model; default_modelkwargs...)
    return nothing
end

calibration = load("tke-data/tke-scaled-flux-mega-batch.jld2", "tke_calibration")

# Optimal parameters
chain = calibration.markov_chains[end]

c★ = optimal(chain).param

ncases = length(calibration.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=ncases, nrows=2, figsize=(16, 5))

for i = 1:ncases
    nll = calibration.negative_log_likelihood.batch[i]

    f = nll.model.constants.f
    N² = nll.model.bcs.T.bottom.condition * nll.model.constants.α * nll.model.constants.g

    data = nll.data # data *should* be the same for tke and kpp

    model = nll.model
    loss = nll.loss

    ji = loss.targets[1]
    jf = loss.targets[end]

    initialize_and_run_until!(model, data, c★, ji, jf)

    if i == 1
        datalbl = "LES"
        modellbl = "TKE-based model"
        datakwargs = merge(default_datakwargs, Dict(:label=>datalbl))
        modelkwargs = merge(default_modelkwargs, Dict(:label=>modellbl))
    else
        datakwargs = default_datakwargs
        modelkwargs = default_modelkwargs
    end

    # Temperature field
    ax = axs[1, i]

    plot_model_field!(ax, :T, model, modelkwargs)
     plot_data_field!(ax, :T, data, ji, jf, datakwargs)

    if i == 1
        leg = legend(markerfirst=false, loc=6, bbox_to_anchor=(-0.6, 0.7, 1.0, 0.25), 
                     prop=Dict(:size=>10))
        leg.set_zorder(1)

        text(0.95, 1.05, L"T", transform=ax.transAxes, fontsize=fs, horizontalalignment="center")
    end

    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    ax.ticklabel_format(useOffset=false)
    removespines("top", "right", "left", "bottom")
    strat = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))

    lims = ax.get_ylim()
    ylim(3*lims[1]/4, 0.01*(lims[2]-lims[1]))


    # Velocity fields
    ax = axs[2, i]
     plot_data_field!(ax, :U, data, ji, jf, default_datakwargs)
    plot_model_field!(ax, :U, model, default_modelkwargs)

    f != 0 && plot_model_field!(ax, :V, model, thin(default_modelkwargs))
    f != 0 &&  plot_data_field!(ax, :V, data, ji, jf, thin(default_datakwargs))

    if i == 1
        text(0.95, 0.85, L"U", transform=ax.transAxes, fontsize=fs, horizontalalignment="center")
        text(-0.1, 1.0, L"V", transform=ax.transAxes, fontsize=fs, horizontalalignment="center")
    end

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")

    if i == 1
        text(0.5, -0.2, 
             @sprintf("\$ N^2 = 10^{%d} \\, \\mathrm{s^{-2}}\$", log10(N²)),
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")
    else
        text(0.5, -0.2,
             @sprintf("\$ 10^{%d} \$", log10(N²)),
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")
    end

    lims = ax.get_ylim()
    ylim(3*lims[1]/4, 0.01*(lims[2]-lims[1]))
end

pause(0.1)

xshift = 0.0
yshift = 0.01
for ax in axs
    pos = get_position(ax)
    pos[1] += xshift
    pos[2] += yshift
    ax.set_position(pos)
end

savefig("mega-batch-viz.png", dpi=480)
