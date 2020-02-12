include("setup.jl")
include("utils.jl")

using JLD2

filename = "multi-calibration-equilibrium-10000.jld2"

file = jldopen(filename)

@show file

close(file)
