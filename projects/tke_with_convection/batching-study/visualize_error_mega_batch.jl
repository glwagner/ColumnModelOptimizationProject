using ColumnModelOptimizationProject, PyCall

include("../setup.jl")
include("../utils.jl")

axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
make_axes_locatable = axes_grid1.make_axes_locatable
rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")
rc("axes", labelsize=10, titlesize=10)
rc("xtick", labelsize=10)
rc("ytick", labelsize=10)

 default_datakwargs = Dict(:linewidth=>3, :alpha=>0.35, :linestyle=>"-", :color=>"k")
default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.9, :linestyle=>"--", :color=>defaultcolors[1])

get_position(ax) = [b for b in ax.get_position().bounds]

function temperature_discrepency(model, data, c★, ji, jf)
    initialize_and_run_until!(model, data, c★, ji, jf)

    grid = model.grid
    δ = CellField(grid)
    θ² = CellField(data.grid)
    T² = CellField(model.grid)

    set!(δ, data.T[jf])

    for i in eachindex(δ)
        @inbounds δ[i] = (δ[i] - model.solution.T[i])^2
        @inbounds T²[i] = model.solution.T[i]^2

    end

    for i in eachindex(T²)
        @inbounds θ²[i] = data.T[jf][i]^2
    end

    return mean(δ) / mean(θ²)
end

tke_calibration = load("tke-data/tke-scaled-flux-mega-batch.jld2", "calibration")
kpp_calibration = load("kpp-data/kpp-mega-batch.jld2", "kpp_calibration")

# Optimal parameters
tke_chain = tke_calibration.markov_chains[end]
kpp_chain = kpp_calibration.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
@show kpp_defaults = kpp_calibration.markov_chains[1][1].param

ncases = length(calibration.negative_log_likelihood.batch)

tke_error = zeros(4, 2)
kpp_error = zeros(4, 2)
kpp_default_error = zeros(4, 2)

for i = 1:ncases
    tke_nll = tke_calibration.negative_log_likelihood.batch[i]
    kpp_nll = kpp_calibration.negative_log_likelihood.batch[i]

    f = tke_nll.model.constants.f
    N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

    @show i f N²

    data = tke_nll.data # data *should* be the same for tke and kpp

    tke_model = tke_nll.model
    tke_loss = tke_nll.loss

    kpp_model = kpp_nll.model
    kpp_default_model = deepcopy(kpp_nll.model)
    kpp_loss = kpp_nll.loss

    ji = tke_loss.targets[1]
    jf = tke_loss.targets[end]

    tke_error[i] = temperature_discrepency(tke_model, data, tke_c★, ji, jf)
    kpp_error[i] = temperature_discrepency(kpp_model, data, kpp_c★, ji, jf)
    kpp_default_error[i] = temperature_discrepency(kpp_default_model, data, kpp_defaults, ji, jf)
end

close("all")
fig, axs = subplots(ncols=3, figsize=(7.4, 2.9))

maxerr = max(
             maximum(tke_error),
             maximum(kpp_error),
             maximum(kpp_default_error)
            )

minerr = min(
             minimum(tke_error),
             minimum(kpp_error),
             minimum(kpp_default_error)
            )

sca(axs[1])
title("TKE-based model")
imshow(log10.(tke_error), vmin=log10(minerr), vmax=log10(maxerr))

sca(axs[2])
title("KPP (calibrated)")
imshow(log10.(kpp_error), vmin=log10(minerr), vmax=log10(maxerr))

sca(axs[3])
title("KPP (default)")
im_kpp = imshow(log10.(kpp_default_error), vmin=log10(minerr), vmax=log10(maxerr))

for ax in axs
    sca(ax)
    removespines("left", "top", "right", "bottom")

    yticks((0, 1, 2, 3), ("\$ N^2 = 10^{-7} \$",
                          "\$ N^2 = 10^{-6} \$",
                          "\$ N^2 = 10^{-5} \$",
                          "\$ N^2 = 10^{-4} \$"))

    xticks((0, 1), ("rotating", "non- \n rotating"))

    xlim(-0.5, 1.5)
    ylim(-0.5, 3.5)
end

axs[1].tick_params(left=false, bottom=false)
axs[2].tick_params(left=false, labelleft=false, bottom=false)
axs[3].tick_params(left=false, labelleft=false, bottom=false)

divider = make_axes_locatable(axs[1])
cax_c = divider.append_axes("right", size="15%", pad=0.2)

divider = make_axes_locatable(axs[2])
cax_e = divider.append_axes("right", size="15%", pad=0.2)

divider = make_axes_locatable(axs[3])
cax = divider.append_axes("right", size="15%", pad=0.2)
cbar = colorbar(im_kpp, cax=cax)
cbar.ax.set_ylabel(
    L"\log_{10} \left \langle (T-\bar \theta)^2 \right \rangle / \left \langle \bar \theta^2 \right \rangle")

for cax in (cax_c, cax_e)
    cax.set_facecolor("None")
    for side in ("top", "bottom", "left", "right")
        cax.spines[side].set_visible(false)
    end
    cax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
end

for side in ("top", "bottom", "left", "right")
    cax.spines[side].set_visible(false)
end

xshift = -0.08
yshift = 0.03
for (i, ax) in enumerate(axs)
    pos = get_position(ax)
    pos[1] += (i-1)*xshift
    pos[2] += yshift
    ax.set_position(pos)
end
