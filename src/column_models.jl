struct ColumnData{T, G, C, F, N}
           Fb :: T
           Fu :: T
    bottom_Bz :: T
            κ :: T
            ν :: T
         grid :: G
    constants :: C
            U :: Array{F, 1}
            V :: Array{F, 1}
            T :: Array{F, 1}
            S :: Array{F, 1}
            t :: Array{T, 1}
    initial :: Int
    targets :: NTuple{N, Int}
end

"""
    ColumnData(datapath)

Construct ColumnData, a time-series of 1D profiles from observations or LES,
from a standardized dataset.
"""
function ColumnData(datapath; initial=1, targets=(2, 3, 4), reversed=false)

    constants_dict = Dict()
    for c in (:α, :g, :f, :ρ₀, :cP)
        try
            constants_dict[c] = getconstant("$c", datapath)
        catch
        end
    end
    constants = Constants(; constants_dict...)

    Fb = getbc("Fb", datapath)
    Fu = getbc("Fu", datapath)
    bottom_Bz = getbc("Bz", datapath)

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    κ = getconstant("κ", datapath)
    ν = getconstant("ν", datapath)

    iters = iterations(datapath)
    U = [ CellField(getdata("U", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    V = [ CellField(getdata("V", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    T = [ CellField(getdata("T", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]
    S = [ CellField(getdata("S", datapath, i; reversed=reversed), grid) for i in 1:length(iters) ]

    t = times(datapath)

    ColumnData(Fb, Fu, bottom_Bz, κ, ν, grid, constants, U, V, T, S, t, initial, targets)
end

target_times(cd::ColumnData) = [cd.t[i] for i in cd.targets]
initial_time(cd::ColumnData) = cd.t[cd.initial]

struct ColumnModel{M, T}
    model :: M
    Δt :: T
end
