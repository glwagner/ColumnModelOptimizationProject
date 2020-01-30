#####
##### ColumnData
#####

struct ColumnData{FT, F, G, C, UU, VV, TT, SS}
     fluxes :: F
       grid :: G
  constants :: C
          κ :: FT
          ν :: FT
          U :: UU
          V :: VV
          T :: TT
          S :: SS
          t :: Vector{FT}
    initial :: Int
    targets :: Vector{Int}
end

"""
    ColumnData(datapath)

Construct ColumnData, a time-series of 1D profiles from observations or LES,
from a standardized dataset.
"""
function ColumnData(datapath; initial=1, targets=(2, 3, 4), reversed=false, FT=Float64)

    constants_dict = Dict()

    file = jldopen(datapath, "r")
    constants_dict[:α] = file["buoyancy/equation_of_state/α"]
    constants_dict[:β] = file["buoyancy/equation_of_state/β"]
    constants_dict[:g] = file["buoyancy/gravitational_acceleration"]
    constants_dict[:f] = file["coriolis/f"]
    close(file)

    constants = Constants(FT; constants_dict...)

    Qᶿ = get_parameter(datapath, "boundary_conditions", "Qᶿ")
    Qˢ = get_parameter(datapath, "boundary_conditions", "Qˢ")
    Qᵘ = get_parameter(datapath, "boundary_conditions", "Qᵘ")
    Qᵛ = get_parameter(datapath, "boundary_conditions", "Qᵛ")

    dθdz = get_parameter(datapath, "initial_conditions", dθdz)

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    ν = get_parameter(datapath, "closure", "ν")
    κ = get_parameter(datapath, "closure", "κ")

    iters = iterations(datapath)

    U = [ CellField(get_field("U", datapath, i), grid) for i in iters ]
    V = [ CellField(get_field("V", datapath, i), grid) for i in iters ]
    T = [ CellField(get_field("T", datapath, i), grid) for i in iters ]
    S = nothing

    try
        S = [ CellField(get_field("S", datapath, i), grid) for i in iters ]
    catch end

    t = times(datapath)

    return ColumnData((Qᶿ=Qᶿ, Qˢ=Qˢ, Qᵘ=Qᵘ, Qᵛ=Qᵛ), grid, constants, κ, ν, 
                      U, V, T, S, t, initial, targets)
end

target_times(cd::ColumnData) = [cd.t[i] for i in cd.targets]
initial_time(cd::ColumnData) = cd.t[cd.initial]

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
