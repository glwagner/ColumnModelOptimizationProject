using ColumnModelOptimizationProject

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
fontsize = 8

include("../setup.jl")
include("../utils.jl")

datakwargs = Dict(:linewidth=>3, :alpha=>0.4, :linestyle=>"-", :color=>"k")
modelkwargs = Dict(:linewidth=>2, :alpha=>0.8, :linestyle=>"--", :color=>defaultcolors[1])

function thin(kwargs)
    new_kwargs = deepcopy(kwargs)
    new_kwargs[:linewidth] = kwargs[:linewidth]/2
    return new_kwargs
end

function plot_field!(ax, fieldname, model, data, ji, jf, datakwargs, modelkwargs)


    ϕ_model = getproperty(model.solution, fieldname)
    ϕ_data_i = getproperty(data, fieldname)[ji]
    ϕ_data_f = getproperty(data, fieldname)[jf]
    
    sca(ax)

    # Plot data and model
    #plot(ϕ_data_i; linewidth=1, alpha=0.4, linestyle="-", color="k")
    plot(ϕ_data_f; datakwargs...)
    plot(ϕ_model; modelkwargs...)

    return nothing
end

annealing = load("mega-batch.jld2", "annealing")

# Optimal parameters
chain = annealing.markov_chains[end]
c★ = optimal(chain).param

ncases = length(annealing.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=ncases, nrows=2, figsize=(16, 8))

for i = 1:ncases
    nll = annealing.negative_log_likelihood.batch[i]
    model = nll.model
    data = nll.data
    loss = nll.loss
    f = nll.model.constants.f
    N² = nll.model.bcs.T.bottom.condition * nll.model.constants.α * nll.model.constants.g

    ji = loss.targets[1]
    jf = loss.targets[end]
    initialize_and_run_until!(model, data, c★, ji, jf)

    ax = axs[1, i]
    plot_field!(ax, :T, model, data, ji, jf, datakwargs, modelkwargs)

    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    ax.ticklabel_format(useOffset=false)
    removespines("top", "right", "left", "bottom")
    title1 = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))

    ax = axs[2, i]
    plot_field!(ax, :U, model, data, ji, jf, datakwargs, modelkwargs)
    f != 0 && plot_field!(ax, :V, model, data, ji, jf, thin(datakwargs), thin(modelkwargs))

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")

end
