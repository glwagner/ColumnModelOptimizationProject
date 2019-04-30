module KPPOptimization

export
    DefaultFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,

    simple_flux_model

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

function simple_flux_model(datapath::AbstractString)
    data_params, constants_dict = getdataparams(datapath)
    constants = KPP.Constants(; constants_dict...)
    simple_flux_model(constants; data_params...)
end

#=
function simple_flux_model(data::ColumnModelData)
    constants = KPP.Constants(; α=data.α, g=data.g, f=data.f)
    setup = Dict((p, getproperty(data, p)) for p in (:N, :L, :Bz, :Fb, :Fu))
    return simple_flux_model(constants; setup...)
end
=#

function compare_with_data(datapath, iter₀=2, compare_at=[12, 22, 32], parameters=KPP.Parameters(),
                     Δt=10*minute)
    model = simple_flux_model(datapath)
    model.parameters = parameters

    alltimes = times(datapath)
    compare_times = [alltimes[i] for i in compare]
    initial_time = alltimes[iter₀]

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    fields = [:U, :V, :T, :S]
    data_fields = Dict((fld, CellField(grid)) for fld in fields)

    for fld in fields
        OceanTurb.set!(data_fields[fld], getdata(fld, datapath, iter₀))
        setproperty!(model.solution, fld, initial_data[fld])
    end

    solution_dict(model) = Dict((fld, deepcopy(getproperty(model.solution, fld))) for fld in fields)

    solution = Dict(iter₀ => solution_dict(model))
    data = Dict(iter₀ => deepcopy(data_fields))
    t = Dict(iter₀ => initial_time)

    for (i, ti) in enumerate(compare_times)
        idata = compare[i]
        run_until!(model, Δt, ti)

        t[idata] = ti
        solution[idata] = solution_dict(model)

        [OceanTurb.set!(data_fields[fld], getdata(fld, datapath, iter₀)) for fld in fields]
        data[idata] = deepcopy(data_fields)
    end

    return solution, data, t
end

end # module
