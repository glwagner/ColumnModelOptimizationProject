function compare_with_data(datapath; N=nothing, initial_idata=2, idata=[12, 22, 32], parameters=KPP.Parameters(), Δt=10*minute)
    model = simple_flux_model(datapath, N=N)
    model.parameters = parameters

    alltimes = times(datapath)
    compare_times = [alltimes[i] for i in idata]
    initial_time = alltimes[initial_idata]

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    fields = [:U, :V, :T, :S]
    datafields = Dict((fld, CellField(grid)) for fld in fields)

    # Set initial condition
    for fld in fields
        OceanTurb.set!(datafields[fld], getdata(fld, datapath, initial_idata))
        OceanTurb.set!(getproperty(model.solution, fld), datafields[fld])
    end

    solution_dict(model) = Dict((fld, deepcopy(getproperty(model.solution, fld))) for fld in fields)

    # Store results in a dict.
    modeloutput = [solution_dict(model)]
    data = [deepcopy(datafields)]

    for (i, ti) in enumerate(compare_times)
        run_until!(model, Δt, ti)

        push!(modeloutput, solution_dict(model))

        # Load and store data at the points of comparison.
        [ OceanTurb.set!(datafields[fld], getdata(fld, datapath, idata[i])) for fld in fields ]
        push!(data, deepcopy(datafields))
    end

    t = cat([initial_time], compare_times, dims=1)
    i = cat([initial_idata], idata, dims=1)

    return modeloutput, data, t, i
end

function visualize_compare_with_data(filepath; kwargs...)

    model, data, tout, idata = compare_with_data(filepath; kwargs...)

    font_manager = pyimport("matplotlib.font_manager")
    defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    modelgrid = model[1][:U].grid
    datagrid = data[1][:U].grid

    Edata = CellField(datagrid)
    Emodel = CellField(modelgrid)
    Edata_coarse = CellField(modelgrid)

    fig, axs = subplots(ncols=2, figsize=(12, 4))

    removespines("top", "right", ax=axs[1])
    removespines("top", "right", "left", ax=axs[2])
    axs[2].tick_params(left=false, labelleft=false)

    for (i, t) in enumerate(tout)
        Umodel = model[i][:U]
        Vmodel = model[i][:V]
        set!(Emodel, 0.5*sqrt.(Umodel.data.^2 + Vmodel.data.^2))

        Udata = data[i][:U]
        Vdata = data[i][:V]
        set!(Edata, 0.5*sqrt.(Udata.data.^2 + Vdata.data.^2))

        Tmodel = model[i][:T]
        Tdata = data[i][:T]

        sca(axs[1])
        plot(Tdata, "-", color=defaultcolors[i], label=@sprintf("\$ t = %.2f \$ days", t/day), linewidth=4, alpha=0.4)
        plot(Tmodel, "--", color=defaultcolors[i], alpha=0.6, linewidth=2)
        xlabel(L"T \, \mathrm{(K)}")
        ylabel(L"z \, \mathrm{(m)}")

        sca(axs[2])
        plot(Edata, "-", color=defaultcolors[i], label=@sprintf("\$ t = %.2f \$ days", t/day), linewidth=4, alpha=0.4)
        plot(Emodel, "--", color=defaultcolors[i], alpha=0.6, linewidth=2)
        xlabel(L"\frac{1}{2} \left ( U^2 + V^2 \right ) \, \mathrm{(m^2 \, s^{-2})}")
    end

    sca(axs[1])
    legend(loc=4, fontsize=10)

    sca(axs[2])
    legend(loc=4, fontsize=10)

    return fig, axs
end
