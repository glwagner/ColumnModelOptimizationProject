#####
##### ColumnModel
#####

struct ColumnModel{M<:AbstractModel, T}
    model :: M
       Δt :: T
end

Base.getproperty(m::ColumnModel, p::Symbol) = getproperty(m, Val(p))
Base.getproperty(m::ColumnModel, ::Val{p}) where p = getproperty(m.model, p)
Base.getproperty(m::ColumnModel, ::Val{:Δt}) = getfield(m, :Δt)
Base.getproperty(m::ColumnModel, ::Val{:model}) = getfield(m, :model)

#####
##### ColumnData
#####

"""
    struct ColumnData{FT, F, G, C, ICS, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as OceanTurb fields.
"""
struct ColumnData{FT, F, G, C, ICS, UU, VV, TT, SS}
    surface_fluxes :: F
    initial_conditions :: ICS
    grid :: G
    constants :: C
    diffusivities :: D
    U :: UU
    V :: VV
    T :: TT
    S :: SS
    t :: Vector{FT}
end

"""
    ColumnData(datapath)

Construct ColumnData from a time-series of Oceananigans LES data.
"""
function ColumnData(datapath; initial=1, targets=(2, 3, 4), reversed=false, FT=Float64)

    # For OceanTurb.Constants
    constants_dict = Dict()

    file = jldopen(datapath, "r")
    constants_dict[:α] = file["buoyancy/equation_of_state/α"]
    constants_dict[:β] = file["buoyancy/equation_of_state/β"]
    constants_dict[:g] = file["buoyancy/gravitational_acceleration"]
    constants_dict[:f] = file["coriolis/f"]
    close(file)

    constants = Constants(FT; constants_dict...)

    # Surface fluxes
    Qᵘ = get_parameter(datapath, "boundary_conditions", "Qᵘ")
    Qᵛ = get_parameter(datapath, "boundary_conditions", "Qᵛ")
    Qᶿ = get_parameter(datapath, "boundary_conditions", "Qᶿ")
    Qˢ = get_parameter(datapath, "boundary_conditions", "Qˢ")

    # Bottom temperature and salinity gradient
    dTdz = get_parameter(datapath, "initial_conditions", dθdz)
    dSdz = 0.0 #get_parameter(datapath, "initial_conditions", dsdz)

    # Grid
    N, L = get_grid_params(datapath)
    grid = UniformGrid(N, L)

    background_ν = get_parameter(datapath, "closure", "ν")
    background_κ = get_parameter(datapath, "closure", "κ")

    iters = iterations(datapath)

    U = [ CellField(get_field("U", datapath, i), grid) for i in iters ]
    V = [ CellField(get_field("V", datapath, i), grid) for i in iters ]
    T = [ CellField(get_field("T", datapath, i), grid) for i in iters ]
    S = nothing

    try
        S = [ CellField(get_field("S", datapath, i), grid) for i in iters ]
    catch end

    t = times(datapath)

    return ColumnData((Qᶿ=Qᶿ, Qˢ=Qˢ, Qᵘ=Qᵘ, Qᵛ=Qᵛ), (dTdz=dTdz, dSdz=dSdz), 
                      grid, constants, (ν=background_ν, κ=(T=background_κ, S=background_κ)), 
                      initial, targets, U, V, T, S, t)
end

target_times(cd::ColumnData) = [cd.t[i] for i in cd.targets]
initial_time(cd::ColumnData) = cd.t[cd.initial]
