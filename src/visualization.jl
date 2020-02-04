styles = ("--", ":", "-.", "o-", "^--")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

"""
    visualize_realizations(data, model, params...)

Visualize the data alongside several realizations of `column_model`
for each set of parameters in `params`.
"""
function visualize_realizations(column_model, column_data, targets, params::FreeParameters...;
                                     figsize = (10, 4),
                                 paramlabels = ["" for p in params], datastyle="-",
                                 modelkwargs = Dict(),
                                  datakwargs = Dict(),
                                legendkwargs = Dict(),
                                      fields = (:U, :V, :T)
                                )

    # Default kwargs for plot routines
    default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8)
    default_datakwargs = Dict(:linewidth=>3, :alpha=>0.6)
    default_legendkwargs = Dict(:fontsize=>10, :loc=>"best", :frameon=>true, :framealpha=>0.5)

    # Merge defaults with user-specified options
     modelkwargs = merge(default_modelkwargs, modelkwargs)
      datakwargs = merge(default_datakwargs, datakwargs)
    legendkwargs = merge(default_legendkwargs, legendkwargs)

    #
    # Make plot
    #

    fig, axs = subplots(ncols=length(fields), figsize=figsize)

    for (iparam, param) in enumerate(params)
        set!(column_model, param)
        set!(column_model, column_data, targets[1])

        for (iplot, i) in enumerate(targets)
            run_until!(column_model.model, column_model.Δt, column_data.t[i])

            if iplot == 1
                lbl =  @sprintf("%s Model, \$ t = %0.2f \$ hours",
                                paramlabels[iparam], column_data.t[i]/hour)
            else
                lbl = ""
            end

            for (ipanel, field) in enumerate(fields)
                sca(axs[ipanel])
                mfld = getproperty(column_model.model.solution, field)
                plot(mfld, styles[iparam]; color=defaultcolors[iplot],
                     label=lbl, modelkwargs...)
            end
        end
    end

    for (iplot, i) in enumerate(targets)
        lbl = iplot == 1 ? "LES, " : ""
        lbl *= @sprintf("\$ t = %0.2f \$ hours", column_data.t[i]/hour)

        for (ipanel, field) in enumerate(fields)
            sca(axs[ipanel])
            dfld = getproperty(column_data, field)[i]
            plot(dfld, datastyle; label=lbl, color=defaultcolors[iplot], datakwargs...)
        end
    end


    sca(axs[1])
    removespines("top", "right")
    legend(; legendkwargs...)

    for iax in 2:length(axs)-1
        sca(axs[iax])
        removespines("top", "right", "left")
        axs[iax].tick_params(left=false, labelleft=false)
    end

    sca(axs[end])
    axs[end].yaxis.set_label_position("right")
    axs[end].tick_params(left=false, labelleft=false, right=true, labelright=true)
    removespines("top", "left")
    ylabel(L"z \, \mathrm{(meters)}")

    for (i, ax) in enumerate(axs)
        if fields[i] === :U
            sca(ax)
            xlabel("\$ U \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
        end

        if fields[i] === :V
            sca(ax)
            xlabel("\$ V \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
        end

        if fields[i] === :T
            sca(ax)
            xlabel("Temperature (Celsius)")
        end

        if fields[i] === :S
            sca(ax)
            xlabel("Salinity (psu)")
        end

        if fields[i] === :e
            sca(ax)
            xlabel("\$ e \$ \$ \\mathrm{(m^2 \\, s^{-2})} \$")
        end
    end

    return fig, axs
end

function plot_loss_function(loss, model, data, params...; 
                            labels=["Parameter set $i" for i = 1:length(params)],
                            time_norm=:second)

    numerical_time_norm = eval(time_norm)

    fig, axs = subplots()

    for (i, param) in enumerate(params)
        evaluate!(loss, param, model, data)
        plot(loss.time / numerical_time_norm, loss.error, label=labels[i])
    end

    removespines("top", "right")

    time_units = string(time_norm, "s")
    xlabel("Time ($time_units)")
    ylabel("Time-resolved loss function")

    return fig, axs
end

function visualize_loss_function(loss, model, data, params...;
                                 labels=["Parameter set $i" for i = 1:length(params)])


end
