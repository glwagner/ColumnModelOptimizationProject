using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
fontsize = 8

 datakwargs = Dict(:linewidth=>3, :alpha=>0.4, :linestyle=>"-", :color=>"k")
modelkwargs = Dict(:linewidth=>2, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[1])

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

annealing = load("tke-data/tke-mega-batch.jld2", "annealing")

# Optimal parameters
chain = annealing.markov_chains[end]

c★ = optimal(chain).param

ncases = length(annealing.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=ncases, nrows=2, figsize=(16, 8))

for i = 1:ncases
    nll = annealing.negative_log_likelihood.batch[i]

    f = nll.model.constants.f
    N² = nll.model.bcs.T.bottom.condition * nll.model.constants.α * nll.model.constants.g

    data = nll.data # data *should* be the same for tke and kpp

    model = nll.model
    loss = nll.loss

    ji = loss.targets[1]
    jf = loss.targets[end]

    initialize_and_run_until!(model, data, c★, ji, jf)

    ax = axs[1, i]
     plot_data_field!(ax, :T, data, ji, jf, datakwargs)
    plot_model_field!(ax, :T, model, modelkwargs)

    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    ax.ticklabel_format(useOffset=false)
    removespines("top", "right", "left", "bottom")
    title1 = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))

    ax = axs[2, i]
    plot_model_field!(ax, :U, model, modelkwargs)
     plot_data_field!(ax, :U, data, ji, jf, datakwargs)

    f != 0 &&  plot_data_field!(ax, :V, data, ji, jf, thin(datakwargs))
    f != 0 && plot_model_field!(ax, :V, model, thin(modelkwargs))

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")
end
