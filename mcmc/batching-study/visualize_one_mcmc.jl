using ColumnModelOptimizationProject

using PyCall

mplot3d = pyimport("mpl_toolkits.mplot3d")
Axes3D = mplot3d.Axes3D

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")

include("../setup.jl")
include("../utils.jl")

case = "tke-scaled-flux-mega-batch.jld2"
casepath = joinpath("tke-data", case)
calibration = load(casepath, "calibration")

chain = calibration.markov_chains[end]

function plot_pdf(name, chain, color; labels=true, disp=1.2)
    samples = Dao.params(chain, after=1)
    c = map(x -> getproperty(x, name), samples)
    
    c_opt = getproperty(optimal(chain).param, name)
    c_med = median(c)
    c_bar = mean(c)
    
    ρ, _, _ = plt.hist(c, bins=100, alpha=0.6, density=true, facecolor=color)
    
    ρmax = maximum(ρ)

    if labels
        opt_lbl = "optimal"
        med_lbl = "median"
        bar_lbl = "mean"
    else
        opt_lbl = ""
        med_lbl = ""
        bar_lbl = ""
    end
    
    plot(c_opt, disp*ρmax, "*", mec="k", mfc=color, markersize=8, linewidth=0.5, alpha=0.6, 
         label=opt_lbl)
    plot(c_med, disp*ρmax, "o", mec="k", mfc=color, markersize=8, linewidth=0.5, alpha=0.6,
         label=med_lbl)
    plot(c_bar, disp*ρmax, "^", mec="k", mfc=color, markersize=8, linewidth=0.5, alpha=0.6,
         label=bar_lbl)

    return nothing
end


close("all")
fig, axs = subplots(figsize=(10, 3))

plot_pdf(:Cᴸʷ, chain, defaultcolors[1], labels=false)
plot_pdf(:Cᴸᵇ, chain, "xkcd:tomato", disp=1.5)
legend(prop=Dict(:size=>12), loc=3, markerscale=1.5, bbox_to_anchor=(0.37, 0.4, 1, 1))

removespines("left", "top", "right")
axs.tick_params(left=false, labelleft=false)
tight_layout()

text(0.72, 1.7, L"C^\ell_w", size=16, color=defaultcolors[1])
text(2.50, 0.7, L"C^\ell_b", size=16, color="xkcd:tomato")

savefig("parameter-pdf-viz.png", dpi=480)

#=
fig, axs, ρ = visualize_markov_chain!(
chain, parameter_latex_guide=TKEMassFluxOptimization.parameter_latex_guide)

fig.suptitle(replace(case, ['_', '-'] => " "))
tight_layout()

samples = collect_samples(chain)
c★ = optimal(chain).param

#fig = figure()
fig, axs = subplots(nrows=2, ncols=2)
α = 0.1

sca(axs[1])
scatter(samples[3, :], samples[7, :], alpha=α)
plot(c★[3], c★[7], "o", color="r", markersize=10)

sca(axs[2])
scatter(samples[4, :], samples[7, :], alpha=α)
plot(c★[4], c★[7], "o", color="r", markersize=10)

sca(axs[3])
scatter(samples[3, :], samples[4, :], alpha=α)
plot(c★[3], c★[4], "o", color="r", markersize=10)

#sca(axs[3])
#scatter(samples[3, :], samples[4, :], alpha=0.2)


#fig = figure()
#ax = fig.add_subplot(111, projection="3d")
#scatter(samples[3, :], samples[4, :], samples[7, :], alpha=0.2)
=#
