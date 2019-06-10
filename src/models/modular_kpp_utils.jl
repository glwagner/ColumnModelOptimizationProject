function ColumnModel(cd::ColumnData, Δt; kwargs...)

    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu,
                              dBdz=cd.bottom_Bz, kwargs...)

    return ColumnModelOptimizationProject.ColumnModel(model, Δt)
end

"""
    simple_flux_model(constants=KPP.Constants(f=2Ω*sin(45π/180));
                            N=40, L=400, h₀=0, d=0.1L/N, dBdz=0.01, Fb=1e-8, Fu=0,
                            parameters=KPP.Parameters())

Construct a model forced by 'simple', constant atmospheric buoyancy flux `Fb`
and velocity flux `Fu`, with resolution `N`, domain size `L`, and
and initial linear buoyancy gradient `Bz`.
"""
function simple_flux_model(constants=Constants(); N=128, L, dBdz, Fb, Fu,
                             diffusivity = ModularKPP.LMDDiffusivity(),
                             mixingdepth = ModularKPP.LMDMixingDepth(),
                            nonlocalflux = ModularKPP.LMDCounterGradientFlux()
                            )


    model = ModularKPP.Model(N=N, L=L,
           constants = constants,
             stepper = :BackwardEuler,
         diffusivity = diffusivity,
         mixingdepth = mixingdepth,
        nonlocalflux = nonlocalflux
        )

    # Initial condition
    dTdz = dBdz / (model.constants.α * model.constants.g)
    T₀(z) = 20 + dTdz * z
    model.solution.T = T₀

    # Fluxes
    Fθ = Fb / (model.constants.α * model.constants.g)
    model.bcs.U.top = FluxBoundaryCondition(Fu)
    model.bcs.T.top = FluxBoundaryCondition(Fθ)
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
