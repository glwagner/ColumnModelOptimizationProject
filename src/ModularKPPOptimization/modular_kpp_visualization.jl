function visualize_model(model; dt=60, dout=1*hour, tfinal=4*day)

    U, V, T, S = model.solution

    ntot = Int(tfinal/dt)
    nint = Int(dout/dt)
    nout = Int(ntot/nint)

    fig, axs = subplots(ncols=3, figsize=(12, 4))

    sca(axs[1])
    plot(U)
    cornerspines()
    xlabel(L"U")
    ylabel(L"z \, \mathrm{(m)}")

    sca(axs[2])
    plot(V)
    bottomspine()
    xlabel(L"V")

    sca(axs[3])
    plot(T)
    bottomspine()
    xlabel(L"T")

    for i = 1:nout
        iterate!(model, dt, nint)
        U, V, T, S = model.solution

        sca(axs[1])
        plot(U)

        sca(axs[2])
        plot(V)

        sca(axs[3])
        plot(T)
    end

    return fig, axs
end
