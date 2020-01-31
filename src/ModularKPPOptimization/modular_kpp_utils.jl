function ColumnModel(cd::ColumnData, Δt; Δ=nothing, N=nothing, kwargs...)

    if Δ != nothing
        N = ceil(Int, cd.grid.L / Δ)
    end

    model = simple_flux_model(cd.constants; N=N, L=cd.grid.L, 
                              Qᶿ=cd.surface_fluxes.Qᶿ, Qˢ=cd.surface_fluxes.Qˢ,
                              Qᵘ=cd.surface_fluxes.Qᵘ, Qᵛ=cd.surface_fluxes.Qᵛ,
                              dTdz=cd.initial_conditions.dTdz, 
                              dSdz=cd.initial_conditions.dSdz, 
                              kwargs...)

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_flux_model(constants=Constants(); N=128, L, dTdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
                             diffusivity = ModularKPP.LMDDiffusivity(),
                             mixingdepth = ModularKPP.LMDMixingDepth(),
                            nonlocalflux = ModularKPP.LMDCounterGradientFlux(),
                                kprofile = ModularKPP.StandardCubicPolynomial())

Construct a model with `Constants`, resolution `N`, domain size `L`,
bottom temperature gradient `dTdz`, and forced by

    - temperature flux `Qᶿ`
    - salinity flux `Qˢ`
    - x-momentum flux `Qᵘ`
    - y-momentum flux `Qᵛ`.

The keyword arguments `diffusivity`, `mixingdepth`, nonlocalflux`, and `kprofile` set
their respective components of the `OceanTurb.ModularKPP.Model`.
"""
function simple_flux_model(constants=Constants(); N=128, L, dTdz, dSdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
                           T₀=20.0, S₀=35.0,
                             diffusivity = ModularKPP.LMDDiffusivity(),
                             mixingdepth = ModularKPP.LMDMixingDepth(),
                            nonlocalflux = ModularKPP.LMDCounterGradientFlux(),
                                kprofile = ModularKPP.StandardCubicPolynomial()
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
    Tᵢ(z) = T₀ + dTdz * z
    Sᵢ(z) = S₀ + dSdz * z
    model.solution.T = T₀
    model.solution.S = S₀

    # Surface fluxes
    model.bcs.U.top = FluxBoundaryCondition(Qᵘ)
    model.bcs.V.top = FluxBoundaryCondition(Qᵛ)
    model.bcs.T.top = FluxBoundaryCondition(Qᶿ)
    model.bcs.S.top = FluxBoundaryCondition(Qˢ)

    # Bottom gradients
    model.bcs.T.bottom = GradientBoundaryCondition(dTdz)
    model.bcs.S.bottom = GradientBoundaryCondition(dSdz)

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
