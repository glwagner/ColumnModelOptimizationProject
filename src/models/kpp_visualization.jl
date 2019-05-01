"""
    visualize_realization(column_data, [params, column_model])

Visualize the data alongside a realization of `column_model`
for the given `params`. If `column_model` and `params` are not provided,
only the data is visualized.
"""
function visualize_realization(column_data, params=nothing, column_model=nothing)

    fig, axs = subplots(ncols=4, figsize=(16, 4))

    font_manager = pyimport("matplotlib.font_manager")
    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if model != nothing # initialize the model
        model = column_model.model
        kpp_parameters = KPP.Parameters(; dictify(params)...)
        model.parameters = kpp_parameters
        set!(column_model, column_data, column_data.initial)
    end

    i_data = cat([column_data.initial], [j for j in column_data.targets], dims=1)

    for (iplot, i) in enumerate(i_data)
        model != nothing && run_until!(model, column_model.Δt, column_data.t[i])

        for (ipanel, field) in enumerate((:U, :V, :T, :S))
            sca(axs[ipanel])
            dfld = getproperty(column_data, field)[i]

            if model != nothing
                mfld = getproperty(model.solution, field)
                err = absolute_error(mfld, dfld)
                lbl = @sprintf("\$ t = %0.2f \$ d, \$ E = %.2e \$", column_data.t[i]/day, err)
                plot(mfld, "--", linewidth=2, alpha=0.8, color=defaultcolors[iplot])
            else
                lbl = @sprintf("\$ t = %0.2f \$ d", column_data.t[i]/day)
            end

            plot(dfld, linewidth=3, alpha=0.6, color=defaultcolors[iplot], label=lbl)
        end
    end

    for ax in axs
        sca(ax)
        legend(fontsize=10, loc=4, frameon=true, framealpha=0.5)
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
