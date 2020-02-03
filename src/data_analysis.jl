function variance(data, field_name, i)
    field = getproperty(data, field_name)[i]
    field_mean = mean(field.data)

    variance = zero(eltype(field))
    for j in eachindex(field)
        @inbounds variance += (field[j] - field_mean)^2 * Δf(field, j)
    end

    return variance
end

function max_variance(data, field_name, targets=1:length(data.t))
    maximum_variance = 0.0

    for target in targets
        field = getproperty(data, field_name)[target]
        fieldmean = mean(field.data)
        maximum_variance = max(maximum_variance, variance(data, field_name, target))
    end

    return maximum_variance
end

#=
function time_averaged_variance(data, field_name, targets)
    variance_time_series = zeros(length(targets))

    for (i, target) in enumerate(targets)
end
=#

function max_variance(data, loss::LossFunction)
    max_variances = zeros(length(loss.fields))
    for (ifield, field) in enumerate(loss.fields)
        max_variances[ifield] = get_weight(weight, ifield) * max_variance(data, field, loss.targets)
    end
    return max_variances
end


function summarize_data(filepath; figaxs=subplots(ncols=2, sharey=true, figsize=(8, 4)),
                        idata=nothing, title=nothing)
                        
    fig, axs = figaxs

    if title != nothing
        fig.suptitle(title)
    end

    # Setup
    N, L = get_grid_params(filepath)
    grid = UniformGrid(N, L)
    U = CellField(grid)
    V = CellField(grid)
    𝒰 = CellField(grid) # speed
    T = CellField(grid)
    #S = CellField(grid)

    iters = get_iterations(filepath)
    t = get_times(filepath)

    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    font_manager = pyimport("matplotlib.font_manager")

    font = font_manager.FontProperties()
    font.set_style("normal")
    font.set_weight("light")
    font.set_size("large")

    axs[2].tick_params(left=false, labelleft=false)

    if idata === nothing
        niters = length(iters)
        idata = [1, round(Int, (niters-1)/2)+1, length(iters)]
    end

    for (iplot, i) = enumerate(idata)

        iter = iters[i]

        tlabel = @sprintf("\$ t = %.1f \$ days", t[i]/day)

        OceanTurb.set!(U, get_data("U", filepath, iter))
        OceanTurb.set!(V, get_data("V", filepath, iter))
        OceanTurb.set!(T, get_data("T", filepath, iter))
        #OceanTurb.set!(S, get_data("S", filepath, iter))

        OceanTurb.set!(𝒰, sqrt.(U.data.^2 + V.data.^2))

        sca(axs[1])
        plot(T, label=tlabel, color=defaultcolors[iplot])
        removespines("top", "right")
        xlabel(L"T")
        ylabel("\$ z \$ (m)")
        legend(fontsize=10)

        sca(axs[2])
        plot(U, "--", alpha=0.5, linewidth=1, color=defaultcolors[iplot])
        plot(V, ":", alpha=0.5, linewidth=1, color=defaultcolors[iplot])
        plot(𝒰, "-", label=tlabel, color=defaultcolors[iplot])
        removespines("top", "left", "right")
        xlabel(L"U, V, \, \mathrm{and} \, \sqrt{U^2 + V^2}")

        legend(fontsize=10, loc=4)
    end

    return fig, axs
end
