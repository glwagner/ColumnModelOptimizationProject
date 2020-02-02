function ColumnModel(cd::ColumnData, Δt; Δ=nothing, N=nothing, kwargs...)

    if Δ != nothing
        N = ceil(Int, cd.grid.L / Δ)
    end

    model = simple_flux_model(cd.constants; N=N, L=cd.grid.L, 
                              Qᶿ=cd.surface_fluxes.Qᶿ, Qˢ=cd.surface_fluxes.Qˢ,
                              Qᵘ=cd.surface_fluxes.Qᵘ, Qᵛ=cd.surface_fluxes.Qᵛ,
                              Qᵉ=cd.surface_fluxes.Qᵉ,
                              dTdz=cd.initial_conditions.dTdz, 
                              dSdz=cd.initial_conditions.dSdz, 
                              kwargs...)

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_flux_model(constants=Constants(); N=128, L, dTdz, dSdz,
                           Qᶿ=0.0, Qˢ=0.0, Qᵘ=0.0, Qᵛ=0.0, Qᵉ=0.0, T₀=20.0, S₀=35.0,
                            tke_equation = TKEMassFlux.TKEParameters(),
                           mixing_length = TKEMassFlux.SimpleMixingLength(),
                           nonlocal_flux = nothing,
                           kwargs...)

Construct a model with `Constants`, resolution `N`, domain size `L`,
bottom temperature gradient `dTdz`, and forced by

    - temperature flux `Qᶿ`
    - salinity flux `Qˢ`
    - x-momentum flux `Qᵘ`
    - y-momentum flux `Qᵛ`.

The keyword arguments `diffusivity`, `mixingdepth`, nonlocalflux`, and `kprofile` set
their respective components of the `OceanTurb.TKEMassFlux.Model`.
"""
function simple_flux_model(constants=Constants(); N=128, L, dTdz, dSdz,
                           Qᶿ=0.0, Qˢ=0.0, Qᵘ=0.0, Qᵛ=0.0, Qᵉ=0.0, T₀=20.0, S₀=35.0,
                            tke_equation = TKEMassFlux.TKEParameters(),
                           mixing_length = TKEMassFlux.SimpleMixingLength(),
                           nonlocal_flux = nothing,
                           kwargs...)

    model = TKEMassFlux.Model(N=N, L=L,
             constants = constants,
               stepper = :BackwardEuler,
         mixing_length = mixing_length,
          tke_equation = tke_equation,
         nonlocal_flux = nonlocal_flux
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
    model.bcs.e.top = FluxBoundaryCondition(Qᵉ)

    # Bottom gradients
    model.bcs.T.bottom = GradientBoundaryCondition(dTdz)
    model.bcs.S.bottom = GradientBoundaryCondition(dSdz)

    return model
end

#=
function visualize_model(model; dt=60, dout=1*hour, tfinal=4*day)

    U, V, T, S, e = model.solution

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
=#
