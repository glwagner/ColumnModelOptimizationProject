using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
fontsize = 8

     datakwargs = Dict(:linewidth=>3, :alpha=>0.4, :linestyle=>"-", :color=>"k")
tke_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[1])
kpp_modelkwargs = Dict(:linewidth=>1, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[2])

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

tke_annealing = load("tke-mega-batch.jld2", "annealing")
kpp_annealing = load("kpp-mega-batch.jld2", "annealing")

# Optimal parameters
tke_chain = tke_annealing.markov_chains[end]
kpp_chain = kpp_annealing.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
@show kpp_defaults = kpp_annealing.markov_chains[1][1].param

ncases = length(tke_annealing.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=ncases, nrows=2, figsize=(16, 8))

for i = 1:ncases
    tke_nll = tke_annealing.negative_log_likelihood.batch[i]
    kpp_nll = kpp_annealing.negative_log_likelihood.batch[i]

    f = tke_nll.model.constants.f
    N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

    data = tke_nll.data # data *should* be the same for tke and kpp

    tke_model = tke_nll.model
    tke_loss = tke_nll.loss

    kpp_model = kpp_nll.model
    kpp_loss = kpp_nll.loss

    ji = tke_loss.targets[1]
    jf = tke_loss.targets[end]

    initialize_and_run_until!(tke_model, data, tke_c★, ji, jf)
    initialize_and_run_until!(kpp_model, data, kpp_c★, ji, jf)
    #initialize_and_run_until!(kpp_model, data, kpp_defaults, ji, jf)

    ax = axs[1, i]
     plot_data_field!(ax, :T, data, ji, jf, datakwargs)
    plot_model_field!(ax, :T, tke_model, tke_modelkwargs)
    #plot_model_field!(ax, :T, kpp_model, kpp_modelkwargs)

    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    ax.ticklabel_format(useOffset=false)
    removespines("top", "right", "left", "bottom")
    title1 = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))

    ax = axs[2, i]
    plot_model_field!(ax, :U, tke_model, tke_modelkwargs)
    #plot_model_field!(ax, :U, kpp_model, kpp_modelkwargs)
     plot_data_field!(ax, :U, data, ji, jf, datakwargs)

    f != 0 &&  plot_data_field!(ax, :V, data, ji, jf, thin(datakwargs))
    f != 0 && plot_model_field!(ax, :V, tke_model, thin(tke_modelkwargs))
    #f != 0 && plot_model_field!(ax, :V, kpp_model, thin(kpp_modelkwargs))

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")
end
