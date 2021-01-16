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
## Oceananigans
file = files[1]
filename = LESbrary_path * file
jldfile = jldopen(filename)
keys(jldfile)
keys(jldfile["timeseries"]["T"])
U = []
V = []
T = []
t = []
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
    push!(t, jldfile["timeseries"]["t"][key])
end

T_bottom = (T[1][2] - T[1][1]) / Δz

## OceanTurb
N = length(T[1])        # Model resolution
H = abs(jldfile["grid"]["zF"][1])        # Vertical extent of the model domain
Qᶿ = T_top       # Surface buoyancy flux (positive implies cooling)
dTdz = T_bottom       # Interior/initial temperature gradient
Δt = 1minute
constants = Constants(f=1e-4)

# Build the model with a Backward Euler timestepper
model = TKEMassFlux.Model(              grid = UniformGrid(N=N, H=H), 
                                     stepper = :BackwardEuler,
                                   constants = constants)

# Set initial condition
model.solution.T.data[1:N] .= copy(T[1])

# Set boundary conditions
model.bcs.T.top = FluxBoundaryCondition(Qᶿ)
model.bcs.T.bottom = GradientBoundaryCondition(dTdz)
##
# Run the model
TKE_T = []
push!(TKE_T, model.solution.T.data[1 : N ])
for i in 2:length(t)
    run_until!(model, Δt, t[i])
    push!(TKE_T, model.solution.T.data[1 : N ])
end
##
using GLMakie, Printf

fig = GLMakie.Figure(resolution = (1200, 700))

timeslider = Slider(fig, range = Int.(range(1, length(T), length = length(T))), startvalue = 1)
time_node = timeslider.value


ax1 = fig[1, 1] = Axis(fig, title = "Temperature", titlesize = 50)
state = @lift(T[$time_node])
tke_state = @lift(TKE_T[$time_node])
line1 = GLMakie.lines!(ax1, state, zC, color = :blue, linewidth = 3)
line2 = GLMakie.lines!(ax1, tke_state, zC, color = :red, linewidth = 3)
ax1.xlabel = "Temperature [ᵒC]"
ax1.xlabelsize = 40
ax1.ylabel = "Depth [m]"
ax1.ylabelsize = 40

timestring = @lift(@sprintf("Day %0.1f", t[$time_node] / 86400))
fig[1, end+1] = vgrid!(
    Legend(fig,
    [line1, line2, ],
    ["LES Temperature", "TKE Temperature"]),
    Label(fig, timestring, width = nothing),
    timeslider,
)

display(fig)