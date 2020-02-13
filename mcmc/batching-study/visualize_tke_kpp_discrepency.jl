using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
fontsize = 8

get_position(ax) = [b for b in ax.get_position().bounds]

     datakwargs = Dict(:linewidth=>3, :alpha=>0.4, :linestyle=>"-", :color=>"k")
tke_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[1])
        kpp_modelkwargs = Dict(:linewidth=>2, :alpha=>0.6, :linestyle=>"--", :color=>"xkcd:tomato")
kpp_default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.6, :linestyle=>":", :color=>"xkcd:tomato")

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

tke_calibration = load("tke-data/tke-mega-batch.jld2", "tke_calibration")
kpp_calibration = load("kpp-data/kpp-mega-batch.jld2", "annealing")

data_label = "LES"
tke_label = "TKE-based model"
kpp_label = "KPP (optimized)"
kpp_default_label = "KPP (default)"

# Optimal parameters
tke_chain = tke_calibration.markov_chains[end]
kpp_chain = kpp_calibration.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
@show kpp_defaults = kpp_calibration.markov_chains[1][1].param

ncases = length(tke_calibration.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=3, nrows=2, figsize=(6, 8))

for (j, i) = enumerate([5]) #2, 4, 5, 8])
    tke_nll = tke_calibration.negative_log_likelihood.batch[i]
    kpp_nll = kpp_calibration.negative_log_likelihood.batch[i]

    f = tke_nll.model.constants.f
    N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

    data = tke_nll.data # data *should* be the same for tke and kpp

    tke_model = tke_nll.model
    tke_loss = tke_nll.loss

    kpp_model = kpp_nll.model
    kpp_default_model = deepcopy(kpp_nll.model)
    kpp_loss = kpp_nll.loss

    @show kpp_model.grid
    @show tke_model.grid

    ji = tke_loss.targets[1]
    jf = tke_loss.targets[end]

    initialize_and_run_until!(tke_model, data, tke_c★, ji, jf)
    initialize_and_run_until!(kpp_model, data, kpp_c★, ji, jf)
    initialize_and_run_until!(kpp_default_model, data, kpp_defaults, ji, jf)

    ax = axs[1, j]
    if j == 1
         plot_data_field!(ax, :T, data, ji, jf, merge(datakwargs, Dict(:label=>data_label)))
        plot_model_field!(ax, :T, tke_model, merge(tke_modelkwargs, Dict(:label=>tke_label)))
        plot_model_field!(ax, :T, kpp_model, merge(kpp_modelkwargs, Dict(:label=>kpp_label)))
        plot_model_field!(ax, :T, kpp_default_model, merge(kpp_default_modelkwargs, Dict(:label=>kpp_default_label)))
    else
         plot_data_field!(ax, :T, data, ji, jf, datakwargs)
        plot_model_field!(ax, :T, tke_model, tke_modelkwargs)
        plot_model_field!(ax, :T, kpp_model, kpp_modelkwargs)
        plot_model_field!(ax, :T, kpp_default_model, kpp_default_modelkwargs)
    end

    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    ax.ticklabel_format(useOffset=false)
    removespines("top", "right", "left", "bottom")
    title1 = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))

    ax = axs[2, j]
     plot_data_field!(ax, :U, data, ji, jf, datakwargs)
    plot_model_field!(ax, :U, tke_model, tke_modelkwargs)
    plot_model_field!(ax, :U, kpp_model, kpp_modelkwargs)
    plot_model_field!(ax, :U, kpp_default_model, kpp_default_modelkwargs)

    f != 0 &&  plot_data_field!(ax, :V, data, ji, jf, thin(datakwargs))
    f != 0 && plot_model_field!(ax, :V, tke_model, thin(tke_modelkwargs))
    f != 0 && plot_model_field!(ax, :V, kpp_model, thin(kpp_modelkwargs))
    f != 0 && plot_model_field!(ax, :V, kpp_default_model, thin(kpp_default_modelkwargs))

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")
end

axs[1, 1].tick_params(left=true, labelleft=true)
axs[1, 1].spines["left"].set_visible(true)
sca(axs[1, 1])
ylabel(L"z \, \mathrm{(m)}")

axs[2, 1].tick_params(left=true, labelleft=true)
axs[2, 1].spines["left"].set_visible(true)
sca(axs[2, 1])
ylabel(L"z \, \mathrm{(m)}")


xshift = 0.05
yshift = 0.00
for ax in axs
    pos = get_position(ax)
    pos[1] += xshift
    pos[2] += yshift
    ax.set_position(pos)
end

sca(axs[1, 1])
leg = legend() #markerfirst=false, loc=6, bbox_to_anchor=(-0.6, 0.7, 1.0, 0.25), 
         #prop=Dict(:size=>10))
