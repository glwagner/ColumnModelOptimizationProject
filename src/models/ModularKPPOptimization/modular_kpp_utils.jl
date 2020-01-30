function ColumnModel(cd::ColumnData, Δt; Δ=nothing, N=nothing, kwargs...)

    if Δ != nothing
        N = ceil(Int, cd.grid.L / Δ)
    end

    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu,
                              dBdz=cd.bottom_Bz, N=N, kwargs...)

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_flux_model(constants=Constants(); N=128, L, dTdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
                             diffusivity = ModularKPP.LMDDiffusivity(),
                             mixingdepth = ModularKPP.LMDMixingDepth(),
                            nonlocalflux = ModularKPP.LMDCounterGradientFlux(),
                                kprofile = ModularKPP.Cubic())

Construct a model with `Constants`, resolution `N`, domain size `L`,
bottom temperature gradient `dTdz`, and forced by

    - temperature flux `Qᶿ`
    - salinity flux `Qˢ`
    - x-momentum flux `Qᵘ`
    - y-momentum flux `Qᵛ`.

The keyword arguments `diffusivity`, `mixingdepth`, nonlocalflux`, and `kprofile` set
their respective components of the `OceanTurb.ModularKPP.Model`.
"""
function simple_flux_model(constants=Constants(); N=128, L, dTdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
                             diffusivity = ModularKPP.LMDDiffusivity(),
                             mixingdepth = ModularKPP.LMDMixingDepth(),
                            nonlocalflux = ModularKPP.LMDCounterGradientFlux(),
                                kprofile = ModularKPP.Cubic()
                            )

    model = ModularKPP.Model(N=N, L=L,
           constants = constants,
             stepper = :BackwardEuler,
         diffusivity = diffusivity,
         mixingdepth = mixingdepth,
        nonlocalflux = nonlocalflux,
            kprofile = kprofile
    )

    # Initial condition
    T₀(z) = 20 + dTdz * z
    model.solution.T = T₀

    # Fluxes
    model.bcs.U.top = FluxBoundaryCondition(Qᵘ)
    model.bcs.V.top = FluxBoundaryCondition(Qᵛ)
    model.bcs.T.top = FluxBoundaryCondition(Qᶿ)
    model.bcs.S.top = FluxBoundaryCondition(Qˢ)
    model.bcs.T.bottom = GradientBoundaryCondition(dTdz)

    return model
end

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
