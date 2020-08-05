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

function set!(cm::ColumnModel, freeparams::FreeParameters{N, T}) where {N, T}

    paramnames, paramtypes = get_free_parameters(cm)
    paramdicts = Dict(( ptypename, Dict{Symbol, T}() ) for ptypename in keys(paramtypes))

    # Filter freeparams into their appropriate category
    for pname in propertynames(freeparams)
        for ptypename in keys(paramtypes)
            pname ∈ paramnames[ptypename] && push!(paramdicts[ptypename],
                                                   Pair(pname, getproperty(freeparams, pname))
                                                  )
        end
    end

    # Set new parameters
    for (ptypename, PType) in paramtypes
        params = PType(; paramdicts[ptypename]...)
        setproperty!(cm.model, ptypename, params)
    end

    return nothing
end

#####
##### ColumnData
#####

"""
    struct ColumnData{FT, F, ICS, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as OceanTurb fields.
"""
struct ColumnData{F, ICS, G, C, D, UU, VV, TΘ, SS, EE, TT}
        surface_fluxes :: F
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
    Qᵘ = get_parameter(datapath, "boundary_conditions", "Qᵘ", 0.0)
    Qᵛ = get_parameter(datapath, "boundary_conditions", "Qᵛ", 0.0)
    Qᶿ = get_parameter(datapath, "boundary_conditions", "Qᶿ", 0.0)
    Qˢ = get_parameter(datapath, "boundary_conditions", "Qˢ", 0.0)

    # Bottom temperature and salinity gradient
    dTdz = get_parameter(datapath, "initial_conditions", "dθdz")
    dSdz = 0.0 #get_parameter(datapath, "initial_conditions", "dsdz")

    # Grid
    N, L = get_grid_params(datapath)
    grid = UniformGrid(N, L)

    background_ν = get_parameter(datapath, "closure", "ν")

    background_κ = (T=get_parameter(datapath, "closure/κ", "T"),
                    S=get_parameter(datapath, "closure/κ", "T"))

    iters = get_iterations(datapath)

    U = [ CellField(get_data("U", datapath, iter), grid) for iter in iters ]
    V = [ CellField(get_data("V", datapath, iter), grid) for iter in iters ]
    T = [ CellField(get_data("T", datapath, iter), grid) for iter in iters ]
    e = [ CellField(get_data("E", datapath, iter), grid) for iter in iters ]

    for (i, iter) in enumerate(iters)
        u² = get_data("U²", datapath, iter) 
        v² = get_data("V²", datapath, iter) 
        w² = get_data("W²", datapath, iter) 

        @. e[i].data[1:N] = ( u²[1:N] - U[i][1:N]^2 + v²[1:N] - V[i][1:N]^2 
                                + 1/2 * (w²[1:N] + w²[2:N+1]) ) / 2
    end

    S = nothing

    try
        S = [ CellField(get_data("S", datapath, iter), grid) for iter in iters ]
    catch end

    t = get_times(datapath)

    return ColumnData((Qᶿ=Qᶿ, Qˢ=Qˢ, Qᵘ=Qᵘ, Qᵛ=Qᵛ, Qᵉ=0.0), (dTdz=dTdz, dSdz=dSdz),
                      grid, constants, (ν=background_ν, κ=background_κ),
                      U, V, T, S, e, t)
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

    return ColumnData(data.surface_fluxes,
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
