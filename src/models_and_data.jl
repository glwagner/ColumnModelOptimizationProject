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
    struct ColumnData{FT, F, ICS, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as OceanTurb fields.
"""
struct ColumnData{F, ICS, G, C, D, UU, VV, TΘ, SS, TT}
    surface_fluxes :: F
    initial_conditions :: ICS
    grid :: G
    constants :: C
    diffusivities :: D
    U :: UU
    V :: VV
    T :: TΘ
    S :: SS
    t :: TT
end

"""
    ColumnData(datapath)

Construct ColumnData from a time-series of Oceananigans LES data.
"""
function ColumnData(datapath)

    # For OceanTurb.Constants
    constants_dict = Dict()

    file = jldopen(datapath, "r")
    constants_dict[:α] = file["buoyancy/equation_of_state/α"]
    constants_dict[:β] = file["buoyancy/equation_of_state/β"]
    constants_dict[:g] = file["buoyancy/gravitational_acceleration"]
    constants_dict[:f] = file["coriolis/f"]
    close(file)

    constants = Constants(; constants_dict...)

    # Surface fluxes
    Qᵘ = get_parameter(datapath, "boundary_conditions", "Qᵘ")
    Qᵛ = get_parameter(datapath, "boundary_conditions", "Qᵛ")
    Qᶿ = get_parameter(datapath, "boundary_conditions", "Qᶿ")
    Qˢ = get_parameter(datapath, "boundary_conditions", "Qˢ")

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
    S = nothing

    try
        S = [ CellField(get_data("S", datapath, i), grid) for i in iters ]
    catch end

    t = get_times(datapath)

    return ColumnData((Qᶿ=Qᶿ, Qˢ=Qˢ, Qᵘ=Qᵘ, Qᵛ=Qᵛ), (dTdz=dTdz, dSdz=dSdz),
                      grid, constants, (ν=background_ν, κ=background_κ),
                      U, V, T, S, t)
end

length(cd::ColumnData) = length(cd.t)
