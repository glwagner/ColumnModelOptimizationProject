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
E = []
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
    push!(E, jldfile["timeseries"]["E"][key][1+ghost:total - ghost])
    push!(t, jldfile["timeseries"]["t"][key])
end

T_bottom = (T[1][2] - T[1][1]) / Δz
zF = jldfile["grid"]["zF"]
## OceanTurb
N = length(T[1])        # Model resolution
H = abs(zF[1])        # Vertical extent of the model domain
Qᶿ = T_top       # Surface buoyancy flux (positive implies cooling)
dTdz = T_bottom       # Interior/initial temperature gradient
Δt = 1minute
constants = Constants(f=1e-4)

# Build the model with a Backward Euler timestepper
model = TKEMassFlux.Model(              grid = UniformGrid(N=N, H=H), 
                                     stepper = :BackwardEuler,
                                   constants = constants)

# Set initial condition
model.solution.U.data[1:N] .= copy(U[1])
model.solution.V.data[1:N] .= copy(V[1])
model.solution.T.data[1:N] .= copy(T[1])

# Set boundary conditions
model.bcs.U.top = FluxBoundaryCondition(U_top)
model.bcs.T.top = FluxBoundaryCondition(Qᶿ)
model.bcs.T.bottom = GradientBoundaryCondition(dTdz)
##
# Run the model
TKE_U = []
TKE_V = []
TKE_T = []
TKE_E = []

push!(TKE_U, model.solution.U.data[1 : N ])
push!(TKE_V, model.solution.V.data[1 : N ])
push!(TKE_E, model.solution.e.data[1 : N ])
push!(TKE_T, model.solution.T.data[1 : N ])
for i in 2:length(t)
    run_until!(model, Δt, t[i])
    push!(TKE_U, model.solution.U.data[1 : N ])
    push!(TKE_V, model.solution.V.data[1 : N ])
    push!(TKE_E, model.solution.e.data[1 : N ])
    push!(TKE_T, model.solution.T.data[1 : N ])
end

function getextrema(TKE_E, E)
    up = maximum([maximum(maximum.(TKE_E)) maximum(maximum.(E))])
    down = minimum([minimum(minimum.(TKE_E)) minimum(minimum.(E))])
    return (down, up)
end
##
using GLMakie, Printf

fig = GLMakie.Figure(resolution = (1200, 700))

timeslider = Slider(fig, range = Int.(range(1, length(T), length = length(T))), startvalue = 1)
time_node = timeslider.value

ax1 = fig[1, 1] = Axis(fig, title = "Temperature [ᵒC]", titlesize = 50)
ax2 = fig[1, 2] = Axis(fig, title = "TKE [m²/s²]", titlesize = 50)
ax3 = fig[2, 1] = Axis(fig, title = " ", titlesize = 50)
ax4 = fig[2, 2] = Axis(fig, title = " ", titlesize = 50)

# Temperature
t_state = @lift(T[$time_node])
t_tke_state = @lift(TKE_T[$time_node])
line1 = GLMakie.lines!(ax1, t_state, zC, color = :blue, linewidth = 3)
line2 = GLMakie.lines!(ax1, t_tke_state, zC, color = :red, linewidth = 3)
ax1.xlabel = " "
ax1.xlabelsize = 40
ax1.ylabel = "Depth [m]"
ax1.ylabelsize = 40

# U Velocity
maxU, minU = getextrema(TKE_U, U)
u_state = @lift(U[$time_node])
u_tke_state = @lift(TKE_U[$time_node])
line1 = GLMakie.lines!(ax3, u_state, zC, color = :blue, linewidth = 3)
line2 = GLMakie.lines!(ax3, u_tke_state, zC, color = :red, linewidth = 3)
ax3.xlabel = "U [m/s]"
ax3.xlabelsize = 40
ax3.ylabel = "Depth [m]"
ax3.ylabelsize = 40

# V Velocity
maxV, minV = getextrema(TKE_V, V)
v_state = @lift(V[$time_node])
v_tke_state = @lift(TKE_V[$time_node])
line1 = GLMakie.lines!(ax4, v_state, zC, color = :blue, linewidth = 3, )
line2 = GLMakie.lines!(ax4, v_tke_state, zC, color = :red, linewidth = 3, )
ax4.xlabel = "V [m/s]"
ax4.xlabelsize = 40

# TKE
e_state = @lift(E[$time_node])
e_tke_state = @lift(TKE_E[$time_node])
line1 = GLMakie.lines!(ax2, e_state, zC, color = :blue, linewidth = 3)
line2 = GLMakie.lines!(ax2, e_tke_state, zC, color = :red, linewidth = 3)
ax2.xlabel = " "
ax2.xlabelsize = 40

timestring = @lift(@sprintf("Day %0.1f", t[$time_node] / 86400))
fig[1:2, 3] = vgrid!(
    Legend(fig,
    [line1, line2, ],
    ["LES ", "TKE "]),
    Label(fig, timestring, width = nothing),
    timeslider,
)

hideydecorations!(ax2, grid = false)
hideydecorations!(ax4, grid = false)


minT, maxT = getextrema(TKE_T, T)
minU, maxU = getextrema(TKE_U, U)
minV, maxV = getextrema(TKE_V, V)
minE, maxE = getextrema(TKE_E, E)
xlims!(ax1, (minT, maxT))
xlims!(ax2, (minE, maxE))
xlims!(ax3, (minU, maxU))
xlims!(ax4, (minV, maxV))
display(fig)

##
seconds = 4
fps = 10
frames = round(Int, fps * seconds )
record_interaction = false
if record_interaction
record(fig, pwd() * "/tke_model.mp4"; framerate = fps) do io
    for i = 1:frames
        sleep(1/fps)
        recordframe!(io)
    end
end
end