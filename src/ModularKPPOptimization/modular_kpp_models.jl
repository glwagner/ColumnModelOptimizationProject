function ColumnModel(cd::ColumnData, Δt; Δ=nothing, N=nothing, kwargs...)

    if Δ != nothing
        N = ceil(Int, cd.grid.L / Δ)
    end

    model = simple_kpp_model(cd.constants; N=N, L=cd.grid.L, 
                              Qᶿ=cd.surface_fluxes.Qᶿ, Qˢ=cd.surface_fluxes.Qˢ,
                              Qᵘ=cd.surface_fluxes.Qᵘ, Qᵛ=cd.surface_fluxes.Qᵛ,
                              dTdz=cd.initial_conditions.dTdz, 
                              dSdz=cd.initial_conditions.dSdz, 
                              kwargs...)

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_kpp_model(constants=Constants(); N=128, L, dTdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
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
function simple_kpp_model(constants=Constants(); N=128, L, dTdz, dSdz, Qᶿ, Qˢ, Qᵘ, Qᵛ,
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
