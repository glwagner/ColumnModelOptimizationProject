struct ColumnData{T, G, C, F, TS}
         Fb :: T
         Fu :: T
  bottom_Bz :: T
          κ :: T
          ν :: T
       grid :: G
  constants :: C
          U :: Vector{F}
          V :: Vector{F}
          T :: Vector{F}
          S :: TS
          t :: Vector{T}
    initial :: Int
    targets :: Vector{Int}
end

"""
    ColumnData(datapath)

Construct ColumnData, a time-series of 1D profiles from observations or LES,
from a standardized dataset.
"""
function ColumnData(datapath; initial=1, targets=(2, 3, 4), reversed=false, T=Float64)

    constants_dict = Dict()

    file = jldopen(datapath, "r")
    constants_dict[:ρ₀] = file["eos/ρ₀"]
     constants_dict[:α] = file["eos/βT"]
     constants_dict[:β] = file["eos/βS"]
     constants_dict[:g] = file["constants/g"]
     constants_dict[:f] = file["constants/f"]
    close(file)

    constants = Constants(T; constants_dict...)

    bcs = Dict()

    try
        bcs[:Fb] = T(getbc("Fb", datapath))
        bcs[:Fu] = T(getbc("Fu", datapath))
        bcs[:bottom_Bz] = T(getbc("Bz", datapath))
    catch
        bcs[:Fb] = T(getbc("Fb", "top", datapath))
        bcs[:Fu] = T(getbc("Fu", "top", datapath))
        bcs[:bottom_Bz] = T(getbc("dbdz", "bottom", datapath))
    end

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    file = jldopen(datapath, "r")
    κ = T(file["closure/κ"])
    ν = T(file["closure/ν"])

    iters = iterations(datapath)
    U = [ CellField(getdata("U", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    V = [ CellField(getdata("V", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    T = [ CellField(getdata("T", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]

    S = nothing

    try
        S = [ CellField(getdata("S", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    catch
    end

    t = times(datapath)

    ColumnData(bcs[:Fb], bcs[:Fu], bcs[:bottom_Bz], κ, ν, grid, constants, U, V, T, S, t, initial, targets)
end

target_times(cd::ColumnData) = [cd.t[i] for i in cd.targets]
initial_time(cd::ColumnData) = cd.t[cd.initial]

struct ColumnModel{M<:AbstractModel, T}
    model :: M
       Δt :: T
end

Base.getproperty(m::ColumnModel, p::Symbol) = getproperty(m, Val(p))
Base.getproperty(m::ColumnModel, ::Val{p}) where p = getproperty(m.model, p)
Base.getproperty(m::ColumnModel, ::Val{:Δt}) = getfield(m, :Δt)
Base.getproperty(m::ColumnModel, ::Val{:model}) = getfield(m, :model)
