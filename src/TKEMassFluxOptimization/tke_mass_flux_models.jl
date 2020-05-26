function ColumnModel(cd::ColumnData, Δt; Δz=nothing, N=nothing, kwargs...)

    if Δz != nothing
        N = ceil(Int, cd.grid.H / Δz)
    end

    model = simple_tke_model(cd.constants; 
                                N = N, 
                                H = cd.grid.H,
                               Qᶿ = cd.surface_fluxes.Qᶿ, 
                               Qˢ = cd.surface_fluxes.Qˢ,
                               Qᵘ = cd.surface_fluxes.Qᵘ, 
                               Qᵛ = cd.surface_fluxes.Qᵛ,
                             dTdz = cd.initial_conditions.dTdz, 
                             dSdz = cd.initial_conditions.dSdz, 
                             kwargs...
                            )

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_tke_model(constants=Constants(); N=128, H, dTdz, dSdz,
                           Qᶿ=0.0, Qˢ=0.0, Qᵘ=0.0, Qᵛ=0.0, Qᵉ=0.0, T₀=20.0, S₀=35.0,
                            tke_equation = TKEMassFlux.TKEParameters(),
                           mixing_length = TKEMassFlux.SimpleMixingLength(),
                           nonlocal_flux = nothing,
                           kwargs...)

Construct a model with `Constants`, resolution `N`, domain height `H`,
bottom temperature gradient `dTdz`, and forced by

    - temperature flux `Qᶿ`
    - salinity flux `Qˢ`
    - x-momentum flux `Qᵘ`
    - y-momentum flux `Qᵛ`.

The keyword arguments `diffusivity`, `mixingdepth`, nonlocalflux`, and `kprofile` set
their respective components of the `OceanTurb.TKEMassFlux.Model`.
"""
function simple_tke_model(constants=Constants(); 
                          N = 128, 
                          H, 
                          dTdz, 
                          dSdz,
                          Qᶿ = 0.0, 
                          Qˢ = 0.0, 
                          Qᵘ = 0.0, 
                          Qᵛ = 0.0, 
                          T₀ = 20.0, 
                          S₀ = 35.0,
                          kwargs...
                         )

    model = TKEMassFlux.Model(;     grid = UniformGrid(N=N, H=H),
                               constants = constants,
                                 stepper = :BackwardEuler,
                                  kwargs...
                             )

    # Initial condition
    Tᵢ(z) = T₀ + dTdz * z
    Sᵢ(z) = S₀ + dSdz * z

    model.solution.T = Tᵢ
    model.solution.S = Sᵢ

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
