smoothstep(z, d) = 0.5 * (1 - tanh(z/d))

function ColumnModel(cd::ColumnData, Δt; kwargs...)
    model = simple_flux_model(cd.constants; L=cd.grid.L, Fb=cd.Fb, Fu=cd.Fu,
                                dBdz=cd.bottom_Bz, kwargs...)
    return ColumnModel(model, Δt)
end

"""
    simple_flux_model(constants=KPP.Constants(f=2Ω*sin(45π/180));
                            N=40, L=400, h₀=0, d=0.1L/N, dBdz=0.01, Fb=1e-8, Fu=0,
                            parameters=KPP.Parameters())

Construct a model forced by 'simple', constant atmospheric buoyancy flux `Fb`
and velocity flux `Fu`, with resolution `N`, domain size `L`, and
and initial linear buoyancy gradient `Bz`.
"""
function simple_flux_model(constants=KPP.Constants(f=2Ω*sin(45π/180));
                            N=40, L=400, h₀=0, d=0.1L/N, dBdz=1e-6, Fb=1e-8, Fu=0,
                            parameters=KPP.Parameters())

    model = KPP.Model(N=N, L=L, parameters=parameters, constants=constants, stepper=:BackwardEuler)

    # Initial condition
    dTdz = dBdz / (model.constants.α * model.constants.g)
    T₀(z) = 20 + dTdz * z * smoothstep(z+h₀, d)
    model.solution.T = T₀

    # Fluxes
    Fθ = Fb / (model.constants.α * model.constants.g)
    model.bcs.U.top = FluxBoundaryCondition(Fu)
    model.bcs.T.top = FluxBoundaryCondition(Fθ)
    model.bcs.T.bottom = GradientBoundaryCondition(dTdz)

    return model
end

function simple_flux_model(datapath::AbstractString; N=nothing)
    data_params, constants_dict = getdataparams(datapath)
    constants = KPP.Constants(; constants_dict...)
    if N != nothing
        data_params[:N] = N
    end
    simple_flux_model(constants; data_params...)
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

function save_data(path, model)
    iteration = iter(model)
    jldopen(path, "a+") do file
        file["timeseries/t/$iteration"] = time(model)
        for fld in (:U, :V, :T, :S)
            file["timeseries/$fld/$iteration"] = collect(data(getproperty(model.solution, fld)))
        end
    end
end

function init_data(path, model)
    Fu = OceanTurb.getbc(model, model.bcs.U.top)
    Fb = OceanTurb.getbc(model, model.bcs.T.top) * model.constants.α * model.constants.g
    Bz = OceanTurb.getbc(model, model.bcs.T.bottom) * model.constants.α * model.constants.g

    jldopen(path, "a+") do file
        for gridfield in (:N, :L)
            file["grid/$gridfield"] = getproperty(model.grid, gridfield)
        end

        for c in (:α, :g, :f, :ρ₀, :cP)
            file["constants/$c"] = getproperty(model.constants, c)
        end

        file["constants/ν"] = model.parameters.KU₀
        file["constants/κ"] = model.parameters.KT₀

        file["boundary_conditions/Fu"] = Fu
        file["boundary_conditions/Fb"] = Fb
        file["boundary_conditions/Bz"] = Bz
    end

    save_data(path, model)

    return nothing
end

function generate_data(filepath, model; dt=60, dout=1*hour, tfinal=2*day)
    ntot = Int(tfinal/dt)
    nint = Int(dout/dt)
    nout = Int(ntot/nint)

    init_data(filepath, model)

    for i = 1:nout
        iterate!(model, dt, nint)
        save_data(filepath, model)
    end

    return nothing
end
