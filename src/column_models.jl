struct ColumnData{T, G, C, F, N}
           Fb :: T
           Fu :: T
            κ :: T
            ν :: T
         grid :: G
    constants :: C
            U :: Array{F, 1}
            V :: Array{F, 1}
            T :: Array{F, 1}
            S :: Array{F, 1}
            E :: Array{F, 1}
            t :: Array{T, 1}
    i_initial :: Int
    i_compare :: NTuple{N, Int}
end

"""
    ColumnData(datapath)

Construct ColumnData, a time-series of 1D profiles from observations or LES,
from a standardized dataset.
"""
function ColumnData(datapath; i_initial=2, i_compare=(10, 18, 26))
    iters = iterations(datapath)

    Fb = getbc("Fb", datapath)
    Fu = getbc("Fu", datapath)

     α = getconstant("α", datapath)
     g = getconstant("g", datapath)
     f = getconstant("f", datapath)
    constants = Constants(α=α, g=g, f=f)

    N, L = getgridparams(datapath)
    grid = UniformGrid(N, L)

    κ = ν = getconstant("κ", datapath)

    t = times(datapath)

    U = [ CellField(getdata("U", datapath, i), grid) for i in 1:length(iters) ]
    V = [ CellField(getdata("V", datapath, i), grid) for i in 1:length(iters) ]
    T = [ CellField(getdata("T", datapath, i), grid) for i in 1:length(iters) ]
    S = [ CellField(getdata("S", datapath, i), grid) for i in 1:length(iters) ]

    E = [ CellField(0.5*(U[i].data.^2 .+ V[i].data.^2), grid)  for i in 1:length(iters) ]

    ColumnData(Fb, Fu, κ, ν, grid, constants, U, V, T, S, E, t, i_initial, i_compare)
end

struct ColumnModel{M, T}
    model :: M
    Δt :: T
end
