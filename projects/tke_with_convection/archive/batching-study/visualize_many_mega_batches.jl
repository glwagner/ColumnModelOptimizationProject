using ColumnModelOptimizationProject

include("../setup.jl")
include("../utils.jl")

fontsize = 8
fs = 12

color1 = defaultcolors[1]
color2 = "k"
α = 0.8

rc("text.latex", preamble="\\usepackage{cmbright}")
rc("font", family="sans-serif")

data_kwargs = Dict(:linewidth=>3, :alpha=>0.35, :linestyle=>"-", :color=>"k")
tke_kwargs = Dict(:linewidth=>2, :alpha=>0.9, :linestyle=>"--", :color=>defaultcolors[1])
kpp_kwargs = Dict(:linewidth=>2, :alpha=>0.9, :linestyle=>"--", :color=>"xkcd:tomato")
kpp_default_kwargs = Dict(:linewidth=>1.5, :alpha=>0.7, :linestyle=>"-.", :color=>defaultcolors[2])

get_position(ax) = [b for b in ax.get_position().bounds]

function thin(kwargs)
    new_kwargs = deepcopy(kwargs)
    new_kwargs[:linewidth] = kwargs[:linewidth]/2
    return new_kwargs
end

function plot_data_field!(ax, fieldname, data, ji, jf, data_kwargs)
    sca(ax)
    ϕ_data_i = getproperty(data, fieldname)[ji]
    ϕ_data_f = getproperty(data, fieldname)[jf]
    plot(ϕ_data_f; data_kwargs...)
    return nothing
end

function plot_model_field!(ax, fieldname, model, default_modelkwargs)
    sca(ax)
    ϕ_model = getproperty(model.solution, fieldname)
    plot(ϕ_model; default_modelkwargs...)
    return nothing
end

tke_calibration = load("tke-data/tke-scaled-flux-mega-batch.jld2", "calibration")
kpp_calibration = load("kpp-data/kpp-mega-batch.jld2", "kpp_calibration")

# Optimal parameters
tke_chain = tke_calibration.markov_chains[end]
kpp_chain = kpp_calibration.markov_chains[end]

tke_c★ = optimal(tke_chain).param
kpp_c★ = optimal(kpp_chain).param
kpp_defaults = kpp_calibration.markov_chains[1][1].param

ncases = length(tke_calibration.negative_log_likelihood.batch)

close("all")
fig, axs = subplots(ncols=ncases, nrows=2, figsize=(16, 5))

for i = 1:ncases
    tke_nll = tke_calibration.negative_log_likelihood.batch[i]
    kpp_nll = kpp_calibration.negative_log_likelihood.batch[i]

    f = tke_nll.model.constants.f
    N² = tke_nll.model.bcs.T.bottom.condition * tke_nll.model.constants.α * tke_nll.model.constants.g

    data = tke_nll.data # data *should* be the same for tke and kpp

    tke_model = tke_nll.model
    tke_loss = tke_nll.loss

    kpp_model = kpp_nll.model
    kpp_default_model = deepcopy(kpp_model)
    kpp_loss = kpp_nll.loss

    ji = tke_loss.targets[1]
    jf = tke_loss.targets[end]

    initialize_and_run_until!(tke_model, data, tke_c★, ji, jf)
    initialize_and_run_until!(kpp_model, data, kpp_c★, ji, jf)
    initialize_and_run_until!(kpp_default_model, data, kpp_defaults, ji, jf)

    if i == 1
        global data_kwargs 
        global tke_kwargs 
        global kpp_kwargs
        global kpp_default_kwargs
        datalbl = "LES"
        tkelbl = "TKE-based model"
        kpplbl = "KPP (calibrated)"
        kppdefaultlbl = "KPP (default)"
        data_kwargs = merge(data_kwargs, Dict(:label=>datalbl))
        tke_kwargs = merge(tke_kwargs, Dict(:label=>tkelbl))
        kpp_kwargs = merge(kpp_kwargs, Dict(:label=>kpplbl))
        kpp_default_kwargs = merge(kpp_default_kwargs, Dict(:label=>kppdefaultlbl))
    end

    # Temperature field
    ax = axs[1, i]

    #plot_model_field!(ax, :T, tke_model, tke_kwargs)
    plot_model_field!(ax, :T, kpp_model, kpp_kwargs)
    plot_model_field!(ax, :T, kpp_default_model, kpp_default_kwargs)
     plot_data_field!(ax, :T, data, ji, jf, data_kwargs)

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

    # Velocity fields
    ax = axs[2, i]
    #plot_model_field!(ax, :U, tke_model, tke_kwargs)
    plot_model_field!(ax, :U, kpp_model, kpp_kwargs)
    plot_model_field!(ax, :U, kpp_default_model, kpp_default_kwargs)
     plot_data_field!(ax, :U, data, ji, jf, data_kwargs)

    #f != 0 && plot_model_field!(ax, :V, tke_model, thin(tke_kwargs))
    f != 0 && plot_model_field!(ax, :V, kpp_model, thin(kpp_kwargs))
    f != 0 && plot_model_field!(ax, :V, kpp_default_model, thin(kpp_default_kwargs))
    f != 0 &&  plot_data_field!(ax, :V, data, ji, jf, thin(data_kwargs))

    if i == 1
        text(0.95, 0.85, L"U", transform=ax.transAxes, fontsize=fs, horizontalalignment="center")
        text(-0.1, 1.0, L"V", transform=ax.transAxes, fontsize=fs, horizontalalignment="center")
    end

    ax.ticklabel_format(useOffset=false)
    ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    removespines("top", "right", "left", "bottom")

    if i == 1 
        text(0.5, -0.2, 
             @sprintf("\$ N^2 = 10^{%d} \\, \\, \\mathrm{s^{-2}}\$", log10(N²)),
             color = color1,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")

        text(0.5, -0.32, 
             @sprintf("\$ f = 10^{%d} \\, \\, \\mathrm{s^{-1}}\$", log10(f)),
             color = color1,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")

    elseif i < 5
        text(0.5, -0.2,
             @sprintf("\$ 10^{%d} \$", log10(N²)),
             color = color1,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")
    elseif i == 5
        text(0.5, -0.2,
             @sprintf("\$ N^2 = 10^{%d} \\, \\, \\mathrm{s^{-2}}\$", log10(N²)),
             color = color2, alpha=α,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")

        text(0.5, -0.32,
             @sprintf("\$ f = 0 \$"),
             color = color2, alpha=α,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")
    else
        text(0.5, -0.2,
             @sprintf("\$ 10^{%d} \$", log10(N²)),
             color = color2, alpha=α,
             transform=ax.transAxes, fontsize=10, horizontalalignment="center")
    end
end


f = 1e-4 # [s⁻¹] Coriolis parameter

# Use 'const' so boundary functions work on the GPU.
const ω = 2π/f  # [s] Inertial period
const u★ = 0.01 # [m² s⁻²], friction velocity

x_momentum_flux(x, y, t) = u★ * cos(ω * t)
y_momentum_flux(x, y, t) = u★ * cos(ω * t)


τˣ = BoundaryFunction{:z, Face, Cell}(x_momentum_flux)
τʸ = BoundaryFunction{:z, Cell, Face}(y_momentum_flux)

u_boundary_condition = HorizontallyPeriodicBoundaryCondition(Flux, τˣ)
v_boundary_condition = HorizontallyPeriodicBoundaryCondition(Flux, τʸ)

pause(0.1)

xshift = 0.0
yshift = 0.03
for ax in axs
    lims = ax.get_ylim()
    ylim(3*lims[1]/4, 0.01*(lims[2]-lims[1]))
    pos = get_position(ax)
    pos[1] += xshift
    pos[2] += yshift
    ax.set_position(pos)
end

savefig("mega-batch-viz-kpp-only.png", dpi=480)
