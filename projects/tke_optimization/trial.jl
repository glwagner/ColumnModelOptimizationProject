using OceanTurb, JLD2

LESbrary_path = "/Users/andresouza/Dropbox/greg-andre/BoundaryLayerTurbulenceData/"
files = [
    "free_convection_Qb1.0e-07_Nsq1.0e-05_Nh256_Nz256_statistics.jld2",
    "free_convection_Qb1.0e-07_Nsq2.0e-06_Nh256_Nz256_statistics.jld2",
    "kato_phillips_Nsq1.0e-03_Qu1.0e-04_Nx256_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx256_Nz128_averages.jld2",
    "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx256_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-04_Qu1.0e-04_Nx512_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx256_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-05_Qu1.0e-04_Nx512_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-06_Qu1.0e-04_Nx512_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx256_Nz256_averages.jld2",
    "kato_phillips_Nsq1.0e-07_Qu1.0e-04_Nx512_Nz256_averages.jld2",
    "stress_driven_Nsq1.0e-04_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
    "stress_driven_Nsq1.0e-05_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
    "stress_driven_Nsq1.0e-06_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
    "stress_driven_Nsq1.0e-07_f1.0e-04_Qu4.0e-04_Nh256_Nz256_averages.jld2",
]
##
file = files[1]
filename = LESbrary_path * file
jldfile = jldopen(filename)
keys(jldfile)
keys(jldfile["timeseries"]["T"])
U = []
V = []
T = []
ghost = 1
total = length(jldfile["timeseries"]["T"]["0"])
Δz = jldfile["grid"]["Δz"]
zC = jldfile["grid"]["zC"]

# Boundary Conditions
if "Qᵘ" ∈ keys(jldfile["boundary_conditions"])
    U_top = jldfile["boundary_conditions"]["Qᵘ"]
else
    U_top = 0.0
end
if "Qᶿ" ∈ keys(jldfile["boundary_conditions"])
    T_top = jldfile["boundary_conditions"]["Qᶿ"]
else
    T_top = 0.0
end

for key in keys(jldfile["timeseries"]["T"])
    push!(U, jldfile["timeseries"]["U"][key][1+ghost:total - ghost])
    push!(V, jldfile["timeseries"]["V"][key][1+ghost:total - ghost])
    push!(T, jldfile["timeseries"]["T"][key][1+ghost:total - ghost])
end

T_bottom = (T[1][2] - T[1][1]) / Δz