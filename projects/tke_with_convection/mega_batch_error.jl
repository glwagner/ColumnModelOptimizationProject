using ColumnModelOptimizationProject, PyCall

include("setup.jl")
include("utils.jl")

improve_filename = "tke_batch_calibration_stability_dz2_dt1.jld2" 
vanilla_filename = "tke_batch_calibration_vanilla_dz2_dt1.jld2"

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

    OceanTurb.set!(δ, data.T[jf]) # interpolate to model grid

    for i in eachindex(δ)
        @inbounds δ[i] = (δ[i] - model.solution.T[i])^2
        @inbounds T²[i] = model.solution.T[i]^2
    end

    θbar = mean(data.T[jf])

    for i in eachindex(T²)
        @inbounds θ²[i] = (data.T[jf][i] - θbar)^2
    end

    return mean(δ) / mean(θ²)
end

improve_calibration = load(joinpath("data", improve_filename), "annealing")
vanilla_calibration = load(joinpath("data", vanilla_filename), "annealing")

# Optimal parameters
improve_chain = improve_calibration.markov_chains[end]
vanilla_chain = vanilla_calibration.markov_chains[end]

improve_c★ = optimal(improve_chain).param
vanilla_c★ = optimal(vanilla_chain).param
baseline_c = vanilla_chain[1].param

@show vanilla_defaults = vanilla_calibration.markov_chains[1][1].param

ncases = length(improve_calibration.negative_log_likelihood.batch)

improve_error = zeros(ncases)
vanilla_error = zeros(ncases)
baseline_error = zeros(ncases)

rotation = zeros(ncases)
stratification = zeros(ncases)
stress = zeros(ncases)
buoyancy_flux = zeros(ncases)

for i = 1:ncases
    improve_nll = improve_calibration.negative_log_likelihood.batch[i]
    vanilla_nll = vanilla_calibration.negative_log_likelihood.batch[i]

    # Environmental parameters
    f = improve_nll.model.constants.f
    N² = improve_nll.model.bcs.T.bottom.condition * improve_nll.model.constants.α * improve_nll.model.constants.g

    # Fluxes
    Qᵘ = improve_nll.model.bcs.U.top.condition
    Qᵛ = improve_nll.model.bcs.V.top.condition
    Qᶿ = improve_nll.model.bcs.T.top.condition
    Qˢ = improve_nll.model.bcs.S.top.condition

    τ = sqrt(Qᵘ^2 + Qᵛ^2)

    g = improve_nll.model.constants.g
    α = improve_nll.model.constants.α
    β = improve_nll.model.constants.β

    Qᵇ = g * (α * Qᶿ - β * Qˢ)

    @show i f N² τ Qᵇ

    rotation[i] = f
    stratification[i] = N²
    stress[i] = τ
    buoyancy_flux[i] = Qᵇ

    data = improve_nll.data # data should be the same for improved and vanilla models

    improve_model = improve_nll.model
    improve_loss = improve_nll.loss

    vanilla_model = vanilla_nll.model
    vanilla_loss = vanilla_nll.loss

    ji = improve_loss.targets[1]
    jf = improve_loss.targets[end]

    improve_error[i] = temperature_discrepency(improve_model, data, improve_c★, ji, jf)
    vanilla_error[i] = temperature_discrepency(vanilla_model, data, vanilla_c★, ji, jf)
    baseline_error[i] = temperature_discrepency(vanilla_model, data, baseline_c, ji, jf)
end

close("all")
fig, axs = subplots(figsize=(20, 8))

plot(improve_error, "o-", alpha=0.6, label="Ri-dependent diffusivities")

plot(vanilla_error, "s-", alpha=0.6, label="Constant Pr diffusivities")

plot(baseline_error, "*-", alpha=0.6, label="Constant Pr diffusivities, stress-driven calibration")

function make_label(f, N², τ, Qᵇ)
    f_lbl = f == 0 ? L"f = 0" : @sprintf("\$ f = 10^{%d} \$", log10(f))
    τ_lbl = τ == 0 ? L"\tau = 0" : @sprintf("\$ \\tau = 10^{%d} \$", log10(τ))

    N²_lbl = @sprintf("\$ N^2 = 10^{%d} \$", log10(N²))
    Qᵇ_lbl = Qᵇ == 0 ? L"Q^b = 0" : @sprintf("\$ Q^b = 10^{%d} \$", log10(Qᵇ))

    return "$f_lbl \n $τ_lbl \n $N²_lbl \n $Qᵇ_lbl"
end

xticks(0:ncases-1, make_label.(rotation, stratification, stress, buoyancy_flux))

axs.set_yscale("log")
removespines("top", "right")
grid(axis="y")
legend()

shift_up!(axs, 0.1)

#=
sca(axs[1])
title("rotating")
imshow(log10.(rotating_compared_error), vmin=log10(minerr), vmax=log10(maxerr), cmap="Reds")

sca(axs[2])
title("non-rotating")
img = imshow(log10.(non_rotating_compared_error), vmin=log10(minerr), vmax=log10(maxerr), cmap="Reds")

for ax in axs
    sca(ax)
    removespines("left", "top", "right", "bottom")

    yticks((0, 1, 2, 3), ("\$ N^2 = 10^{-7} \$",
                          "\$ N^2 = 10^{-6} \$",
                          "\$ N^2 = 10^{-5} \$",
                          "\$ N^2 = 10^{-4} \$"))

    xticks((0, 1, 2), ("TKE-based \n model",
                       "KPP \n (calibrated)",
                       "KPP \n (default)"))
    xlim(-0.5, 2.5)
    ylim(-0.5, 3.5)
end

axs[1].tick_params(left=false, bottom=false)
axs[2].tick_params(left=false, labelleft=false, bottom=false)

divider = make_axes_locatable(axs[1])
cax_c = divider.append_axes("right", size="10%", pad=0.25)

divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="10%", pad=0.25)
cbar = colorbar(img, cax=cax)
cbar.ax.set_ylabel(
    L"\mathrm{error} = \log_{10} \left \langle (T-\bar \theta)^2 \right \rangle / \left \langle \left ( \bar \theta - \langle \bar \theta \rangle \right )^2 \right \rangle \, \Big |_{t=t_f}",
   labelpad=12.0)

cax_c.set_facecolor("None")
for side in ("top", "bottom", "left", "right")
    cax_c.spines[side].set_visible(false)
end
cax_c.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)

for side in ("top", "bottom", "left", "right")
    cax.spines[side].set_visible(false)
end

xshift = -0.12
yshift = 0.03
for (i, ax) in enumerate(axs)
    pos = get_position(ax)
    pos[1] += (i-1)*xshift
    pos[2] += yshift
    ax.set_position(pos)
end

savefig("error-visualization.png", dpi=480)
=#
