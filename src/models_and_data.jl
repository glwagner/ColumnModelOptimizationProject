#####
##### ColumnModel
#####

mutable struct ColumnModel{M<:AbstractModel, T}
    model :: M
       Δt :: T
end

run_until!(model::ColumnModel, time) = run_until!(model.model, model.Δt, time)

Base.getproperty(m::ColumnModel, p::Symbol) = getproperty(m, Val(p))
Base.getproperty(m::ColumnModel, ::Val{p}) where p = getproperty(m.model, p)
Base.getproperty(m::ColumnModel, ::Val{:Δt}) = getfield(m, :Δt)
Base.getproperty(m::ColumnModel, ::Val{:model}) = getfield(m, :model)

function get_free_parameters(cm::ColumnModel)
    paramnames = Dict()
    paramtypes = Dict()
    for pname in propertynames(cm.model)
        p = getproperty(cm.model, pname)
        if typeof(p) <: OceanTurb.AbstractParameters
            paramnames[pname] = propertynames(p)
            paramtypes[pname] = typeof(p)
        end
    end
    return paramnames, paramtypes
end

function set!(cm::ColumnModel, free_parameters::FreeParameters{N, T}) where {N, T}

    paramnames, paramtypes = get_free_parameters(cm)
    paramdicts = Dict(( ptypename, Dict{Symbol, T}() ) for ptypename in keys(paramtypes))

    for ptypename in keys(paramtypes)

        existing_parameters = getproperty(cm.model, ptypename)

        for pname in propertynames(existing_parameters)

            p = pname ∈ propertynames(free_parameters) ?
                    getproperty(free_parameters, pname) :
                    getproperty(existing_parameters, pname)

            paramdicts[ptypename][pname] = p
        end
    end

    # Set new parameters
    for (ptypename, PType) in paramtypes
        new_parameters = PType(; paramdicts[ptypename]...)
        setproperty!(cm.model, ptypename, new_parameters)
    end

    return nothing
end

#####
##### ColumnData
#####

"""
    struct ColumnData{FT, F, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as OceanTurb fields.
"""
struct ColumnData{F, ICS, G, C, D, UU, VV, TΘ, SS, EE, TT, NN}
   boundary_conditions :: F
    initial_conditions :: ICS
                  grid :: G
             constants :: C
         diffusivities :: D
                     U :: UU
                     V :: VV
                     T :: TΘ
                     S :: SS
                     e :: EE
                     t :: TT
                  name :: NN
end

"""
    ColumnData(datapath)

Construct ColumnData from a time-series of Oceananigans LES data saved at `datapath`.
"""
function ColumnData(datapath)

    # For now, we assume salinity-less LES data.

    # For OceanTurb.Constants
    constants_dict = Dict()

    file = jldopen(datapath, "r")
    constants_dict[:α] = file["buoyancy/equation_of_state/α"]
    constants_dict[:β] = 0.0 #file["buoyancy/equation_of_state/β"]
    constants_dict[:g] = file["buoyancy/gravitational_acceleration"]
    constants_dict[:f] = 0.0

    try
        constants_dict[:f] = file["coriolis/f"]
    catch end

    close(file)

    constants = Constants(; constants_dict...)

    # Surface fluxes
    Qᵘ = get_parameter(datapath, "parameters", "boundary_condition_u_top", 0.0)
    # Qᵛ = get_parameter(datapath, "boundary_conditions", "Qᵛ", 0.0)
    Qᵛ = 0.0
    Qᶿ = get_parameter(datapath, "parameters", "boundary_condition_θ_top", 0.0)
    # Qˢ = get_parameter(datapath, "boundary_conditions", "Qˢ", 0.0)
    # Qˢ = 0.0

    # Bottom temperature and salinity gradient
    # dTdz = get_parameter(datapath, "parameters", "dθdz")
    # println("dTdz $(dTdz)")
    # dSdz = 0.0 #get_parameter(datapath, "initial_conditions", "dsdz")

    # Bottom gradients
    dθdz_bottom = get_parameter(datapath, "parameters", "boundary_condition_θ_bottom", 0.0)
    dudz_bottom = get_parameter(datapath, "parameters", "boundary_condition_u_bottom", 0.0)

    name = get_parameter(datapath, "parameters", "name", "")

    # Grid
    N, L = get_grid_params(datapath)
    grid = UniformGrid(N, L)

    background_ν = get_parameter(datapath, "closure", "ν")

    background_κ = (T=get_parameter(datapath, "closure/κ", "T"),
                    S=get_parameter(datapath, "closure/κ", "T"))

    iters = get_iterations(datapath)

    U = [ CellField(get_data("u", datapath, iter), grid) for iter in iters ]
    V = [ CellField(get_data("v", datapath, iter), grid) for iter in iters ]
    T = [ CellField(get_data("T", datapath, iter), grid) for iter in iters ]
    e = [ CellField(get_data("e", datapath, iter), grid) for iter in iters ]

    # # Retrieve parameters for initial condition
    # global_attributes = Dict()
    # jldopen(datapath, "r") do file
    #     for parameter_name in keys(file["parameters"])
    #         global_attributes[parameter_name] = file["parameters/$parameter_name"]
    #     end
    # end
    # function thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep, dθdz_surface_layer, dθdz_thermocline, dθdz_deep)
    #     if thermocline_type == "linear"
    #         return z -> θ_transition + dθdz_thermocline * (z - z_transition)
    #
    #     elseif thermocline_type == "cubic"
    #         p1 = (z_transition, θ_transition)
    #         p2 = (z_deep, θ_deep)
    #         coeffs = fit_cubic(p1, p2, dθdz_surface_layer, dθdz_deep)
    #         return z -> poly(z, coeffs)
    #
    #     else
    #         @error "Invalid thermocline type: $thermocline"
    #     end
    # end
    #
    # thermocline_type = "linear"
    # if "thermocline_type" in keys(global_attributes)
    #     thermocline_type = global_attributes["thermocline_type"]
    # end
    # θ_thermocline = thermocline_structure_function(
    #                             thermocline_type,
    #                             global_attributes["z_transition"],
    #                             global_attributes["θ_transition"],
    #                             global_attributes["z_deep"],
    #                             global_attributes["θ_deep"],
    #                             global_attributes["dθdz_surface_layer"],
    #                             global_attributes["dθdz_thermocline"],
    #                             global_attributes["dθdz_deep"])
    # Ξ(z) = rand() * exp(z / 8)
    # function initial_temperature(z)
    #
    #     noise = 1e-6 * Ξ(z) * global_attributes["dθdz_surface_layer"] * get_parameter(datapath, "grid", "Lz", 0.0)
    #
    #     if global_attributes["z_transition"] < z <= 0
    #         return global_attributes["θ_surface"] + global_attributes["dθdz_surface_layer"] * z + noise
    #
    #     elseif global_attributes["z_deep"] < z <= global_attributes["z_transition"]
    #         return θ_thermocline(z) + noise
    #
    #     else
    #         return global_attributes["θ_deep"] + global_attributes["dθdz_deep"] * (z - global_attributes["z_deep"]) + noise
    #
    #     end
    # end

    # Uᵢ = U[:,1]
    # Vᵢ = V[:,1]
    # Tᵢ = T[:,1]

    for (i, iter) in enumerate(iters)
        u² = get_data("uu", datapath, iter)
        v² = get_data("vv", datapath, iter)
        w² = get_data("ww", datapath, iter)

        @. e[i].data[1:N] = ( u²[1:N] - U[i][1:N]^2 + v²[1:N] - V[i][1:N]^2
                                + 1/2 * (w²[1:N] + w²[2:N+1]) ) / 2
    end

    S = nothing

    try
        S = [ CellField(get_data("S", datapath, iter), grid) for iter in iters ]
    catch end

    t = get_times(datapath)

    # return ColumnData((Qᶿ=Qᶿ, Qˢ=Qˢ, Qᵘ=Qᵘ, Qᵛ=Qᵛ, Qᵉ=0.0), (dTdz=dTdz, dSdz=dSdz),
    #                   grid, constants, (ν=background_ν, κ=background_κ),
    #                   U, V, T, S, e, t)

    initial_temperature = nothing

    return ColumnData((Qᶿ=Qᶿ, Qᵘ=Qᵘ, Qᵉ=0.0, dθdz_bottom=dθdz_bottom, dudz_bottom=dudz_bottom), (initial_temperature=initial_temperature,),
                      grid, constants, (ν=background_ν, κ=background_κ),
                      U, V, T, S, e, t, name)

end

"""
    ColumnData(data::ColumnData, coarse_grid)

Returns `data::ColumnData` interpolated to `grid`.
"""
function ColumnData(data::ColumnData, grid)

    U = [ CellField(grid) for t in data.t ]
    V = [ CellField(grid) for t in data.t ]
    T = [ CellField(grid) for t in data.t ]
    e = [ CellField(grid) for t in data.t ]
    S = nothing

    for i = 1:length(iters)
        set!(U[i], data.U[i])
        set!(V[i], data.V[i])
        set!(T[i], data.T[i])
        set!(e[i], data.e[i])
    end

    return ColumnData(data.boundary_conditions,
                      data.initial_conditions,
                      grid,
                      data.constants,
                      data.diffusivities,
                      U,
                      V,
                      T,
                      S,
                      e,
                      data.t)
end

length(cd::ColumnData) = length(cd.t)
