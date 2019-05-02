"""
    visualize_realization([params, column_model], column_data)

Visualize the data alongside a realization of `column_model`
for the given `params`. If `column_model` and `params` are not provided,
only the data is visualized.
"""
function visualize_realization(params, column_model, column_data;
                               figsize=(16, 4), modelstyle="--", datastyle="-",
                               modelkwargs=Dict(), datakwargs=Dict(), legendkwargs=Dict()
                               )

    # Default kwargs for plot routines
    default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8)
    default_datakwargs = Dict(:linewidth=>3, :alpha=>0.6)
    default_legendkwargs = Dict(:fontsize=>10, :loc=>"best", :frameon=>true, :framealpha=>0.5)

    # Merge defaults with user-specified options
     modelkwargs = merge(default_modelkwargs, modelkwargs)
      datakwargs = merge(default_datakwargs, datakwargs)
    legendkwargs = merge(default_legendkwargs, legendkwargs)

    font_manager = pyimport("matplotlib.font_manager")
    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if column_model != nothing # initialize the model
        set!(column_model, params)
        set!(column_model, column_data, column_data.initial)
    end

    fields = (:U, :V, :T, :S)

    i_data = cat([column_data.initial], [j for j in column_data.targets], dims=1)

    fig, axs = subplots(ncols=4, figsize=figsize)

    for (iplot, i) in enumerate(i_data)
        column_model != nothing && run_until!(column_model.model, column_model.Δt, column_data.t[i])

        for (ipanel, field) in enumerate(fields)
            sca(axs[ipanel])
            dfld = getproperty(column_data, field)[i]

            if column_model != nothing
                mfld = getproperty(column_model.model.solution, field)
                err = absolute_error(mfld, dfld)
                lbl = @sprintf("\$ t = %0.2f \$ d, \$ E = %.2e \$", column_data.t[i]/day, err)
                plot(mfld, modelstyle; color=defaultcolors[iplot], modelkwargs...)
            else
                lbl = @sprintf("\$ t = %0.2f \$ d", column_data.t[i]/day)
            end

            plot(dfld, datastyle; color=defaultcolors[iplot], label=lbl, datakwargs...)
        end
    end

    for ax in axs
        sca(ax)
        legend(; legendkwargs...)
    end

    axs[2].tick_params(left=false, labelleft=false)
    axs[3].tick_params(left=false, labelleft=false)
    axs[4].tick_params(left=false, labelleft=false, right=true, labelright=true)
    axs[4].yaxis.set_label_position("right")

    sca(axs[1])
    xlabel("\$ U \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
    ylabel(L"z \, \mathrm{(meters)}")
    removespines("top", "right")

    sca(axs[2])
    xlabel("\$ V \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
    removespines("top", "right", "left")

    sca(axs[3])
    xlabel("Temperature (Kelvin)")
    removespines("top", "right", "left")

    sca(axs[4])
    xlabel("Salinity (psu)")
    ylabel(L"z \, \mathrm{(meters)}")
    removespines("top", "left")

    return fig, axs
end

visualize_targets(column_data; kwargs...) = visualize_realization(nothing, nothing, column_data; kwargs...)
