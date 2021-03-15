function ColumnModel(cd::ColumnData, Δt; Δz=nothing, N=nothing, kwargs...)

    if Δz != nothing
        N = ceil(Int, cd.grid.H / Δz)
    end

    model = simple_tke_model(cd.constants;
                                N = N,
                                H = cd.grid.H,
                               Qᶿ = cd.boundary_conditions.Qᶿ,
                               Qᵘ = cd.boundary_conditions.Qᵘ,
                      dθdz_bottom = cd.boundary_conditions.dθdz_bottom,
                      dudz_bottom = cd.boundary_conditions.dudz_bottom,
                             kwargs...
                            )


    # set!(cm, cd, 1)
    set!(model.solution.U, cd.U[1])
    set!(model.solution.V, cd.V[1])
    set!(model.solution.T, cd.T[1])
    cd.S != nothing && set!(cm.model.solution.S, cd.S[1])
    model.clock.time = cd.t[1]

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
                          Qᶿ = 0.0,
                          Qᵘ = 0.0,
                 dθdz_bottom = 0.0,
                 dudz_bottom = 0.0,
                          kwargs...
                         )

    model = TKEMassFlux.Model(;     grid = UniformGrid(N=N, H=H),
                               constants = constants,
                                 stepper = :BackwardEuler,
                                  kwargs...
                             )

    # Surface fluxes
    model.bcs.U.top = FluxBoundaryCondition(Qᵘ)
    model.bcs.T.top = FluxBoundaryCondition(Qᶿ)
    # model.bcs.S.top = FluxBoundaryCondition(Qˢ)

    # Bottom gradients
    model.bcs.T.bottom = GradientBoundaryCondition(dθdz_bottom)
    model.bcs.U.bottom = GradientBoundaryCondition(dudz_bottom)
    # model.bcs.S.bottom = GradientBoundaryCondition(dSdz)

    return model
end
