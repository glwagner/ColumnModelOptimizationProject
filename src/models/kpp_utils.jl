module KPPOptimization

export
    DefaultFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,

    simple_flux_model,
    compare_with_data

using
    ColumnModelOptimizationProject,
    OceanTurb,
    StaticArrays,
    JLD2

import OceanTurb: KPP

#
# Parameter sets
#

struct SensitiveParameters{T} <: FieldVector{4, T}
     CRi :: T
     CNL :: T
     CKE :: T  # Unresolved turbulence parameter
end

struct ShearNeutralParameters{T}
    CSL   :: T  # Surface layer fraction
    Cτ    :: T  # Von Karman constant
    CNL   :: T  # Non-local flux proportionality constant
    CRi   :: T  # Critical bulk Richardson number
    K₀    :: T  # Background diffusivity
end

struct FreeConvectionParameters{T} <: FieldVector{6, T}
     CRi :: T
     CNL :: T
     CKE :: T
      Cτ :: T
    Cb_U :: T
    Cb_T :: T
      K₀ :: T
end

struct ShearUnstableParameters{T} <: FieldVector{12, T}
    CSL   :: T  # Surface layer fraction
    Cτ    :: T  # Von Karman constant
    CNL   :: T  # Non-local flux proportionality constant
    Cunst :: T  # Unstable buoyancy flux parameter for wind-driven turbulence
    Cb_U  :: T  # Buoyancy flux parameter for convective turbulence
    Cτb_U :: T  # Wind stress parameter for convective turbulence
    Cb_T  :: T  # Buoyancy flux parameter for convective turbulence
    Cτb_T :: T  # Wind stress parameter for convective turbulence
    Cd_U  :: T  # Wind mixing regime threshold for momentum
    Cd_T  :: T  # Wind mixing regime threshold for tracers
    CRi   :: T  # Critical bulk Richardson number
    CKE   :: T  # Unresolved turbulence parameter
    K₀    :: T  # Background diffusivity
end

function DefaultFreeParameters(freeparamtype)
    allparams = KPP.Parameters()
    freeparams = (getproperty(allparameters, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, type, freeparams...))
end

#
# Loss functions
#

"""
    temperature_loss(params, model, data; iters=1)

Compute the error between model and data for one iteration of
the `model.turbmodel`.
"""
function temperature_loss(params, model, data; iters=1)
    kpp_parameters = KPP_Parameters(params, data.K₀)
    turbmodel = model.turbmodel
    turbmodel.parameters = kpp_parameters

    # Set initial condition as first non-trivial time-step
    turbmodel.solution.U = data.U[2]
    turbmodel.solution.V = data.V[2]
    turbmodel.solution.T = data.T[2]

    loss = 0.0

    for i = 3:3+iters
        run_until!(turbmodel, model.dt, data.t[i])
        T_model = OceanTurb.data(turbmodel.solution.T)
        loss += mean(data.T[i].^2 .- T_model.^2)
    end

    return loss
end

#
# Models
#

"""
    simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                      parameters=KPP.Parameters())

Construct a model forced by 'simple', constant atmospheric buoyancy flux `Fb`
and velocity flux `Fu`, with resolution `N`, domain size `L`, and
and initial linear buoyancy gradient `Bz`.
"""
function simple_flux_model(constants; N=40, L=400, Bz=0.01, Fb=1e-8, Fu=0,
                           parameters=KPP.Parameters())

    model = KPP.Model(N=N, L=L, parameters=parameters, constants=constants,
                      stepper=:BackwardEuler)

    # Initial condition
    Tz = model.constants.α * model.constants.g * Bz
    T₀(z) = 20 + Tz*z
    model.solution.T = T₀

    # Fluxes
    Fθ = Fb / (model.constants.α * model.constants.g)
    model.bcs.U.top = FluxBoundaryCondition(Fu)
    model.bcs.T.top = FluxBoundaryCondition(Fθ)
    model.bcs.T.bottom = GradientBoundaryCondition(Tz)

    return model
end

function simple_flux_model(datapath::AbstractString; N=nothing)
    data_params, constants_dict = getdataparams(datapath)
    @show constants_dict
    constants = KPP.Constants(; constants_dict...)
    if N != nothing
        data_params[:N] = N
    end
    simple_flux_model(constants; data_params...)
end

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

end # module
