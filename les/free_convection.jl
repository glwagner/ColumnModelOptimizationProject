using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

# Two cases from Van Roekel et al (JAMES, 2018)
parameters = Dict(
    :free_convection => Dict(:Qb=>3.39e-8, :Qu=>0.0,      :f=>1e-4, :N²=>1.96e-5, :tf=>8day, :dt=>5.0),
    :wind_stress     => Dict(:Qb=>0.0,     :Qu=>-9.66e-5, :f=>0.0,  :N²=>9.81e-5, :tf=>4day, :dt=>0.1)
)

# Simulation parameters
case = :free_convection
Nx = 128
Nz = 128
 L = 128

N² = 1.96e-5
Qb = 3.39e-8
 f = 1e-4
tf = 8day
dt = 1.0
αθ, g = 2e-4, 9.81

Qθ = Qb / (g*αθ)
const dθdz = N² / (g*αθ)

# Create boundary conditions.
θbcs = HorizontallyPeriodicBCs(top=BoundaryCondition(Flux, Qθ), 
                               bottom=BoundaryCondition(Gradient, dθdz))

# Halo parameters
const θᵣ = 20.0
const Δμ = 3.0

@inline μ(z, Lz) = 0.02 * exp(-(z+Lz) / Δμ)
@inline θ₀(z) = θᵣ + dθdz * z

@inline Fu(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]
@inline Fθ(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * (Φ.T[i, j, k] - θ₀(grid.zC[k]))

# Instantiate the model
model = Model(      arch = GPU(), 
                       N = (Nx, Nx, Nz), L = (L, L, L),
                     eos = LinearEquationOfState(βT=αθ, βS=0.0),
               constants = PlanetaryConstants(f=f, g=g),
                 closure = AnisotropicMinimumDissipation(Cb=0.0),
                 forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw, FT=Fθ),
                     bcs = BoundaryConditions(T=θbcs)
)

# Set initial condition.
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
uᵢ(x, y, z) = 1e-3 * Ξ(z)
θᵢ(x, y, z) = θᵣ + dθdz * z + 1e-3 * dθdz * model.grid.Lz * Ξ(z)
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, T=θᵢ)

Tavg = HorizontalAverage(model, model.tracers.T)

function plot_average_temperature(model, Tavg)
    T = Array(Tavg(model))
    return lineplot(T[2:end-1], model.grid.zC, height=40, canvas=DotCanvas, 
                    xlim=[θᵣ-dθdz*model.grid.Lz, θᵣ], ylim=[-model.grid.Lz, 0])
end

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.05, Δt=Δt, max_change=1.1, max_Δt=10.0)

#
# Set up output
#

init_bcs(file, model) = file["boundary_conditions"] = model.boundary_conditions

filename = @sprintf("%s_Nx%d_Nz%d", case, Nx, Nz)

u(model) = Array(model.velocities.u.data.parent)
v(model) = Array(model.velocities.v.data.parent)
w(model) = Array(model.velocities.w.data.parent)
T(model) = Array(model.tracers.T.data.parent)
νₑ(model) = Array(model.diffusivities.νₑ.data.parent)
κₑ(model) = Array(model.diffusivities.κₑ.T.data.parent)

fields = merge(model.velocities, (T=model.tracers.T,), (νₑ=model.diffusivities.νₑ, κₑ=model.diffusivities.κₑ.T))
outputs = FieldOutputs(fields)
field_writer = JLD2OutputWriter(model, outputs; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=2GiB, interval=6hour, force=true)
push!(model.output_writers, field_writer)

#
# Set up diagnostics
#

frequency = 10
push!(model.diagnostics, MaxAbsFieldDiagnostic(model.velocities.w, frequency=frequency),
                         AdvectiveCFL(wizard, frequency=frequency),
                         DiffusiveCFL(wizard, frequency=frequency))

# 
# Run the simulation
#

terse_message(model, walltime, Δt) =
    @sprintf(
    "i: %d, t: %.4f hours, Δt: %.3f s, wmax: %.6f ms⁻¹, adv cfl: %.3f, diff cfl: %.3f, wall time: %s\n",
    model.clock.iteration, model.clock.time/3600, Δt, 
    model.diagnostics[1].data[end], model.diagnostics[2].data[end], model.diagnostics[3].data[end],
    prettytime(walltime)
   )

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = Base.@elapsed time_step!(model, 100, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    if model.clock.iteration % 10000 == 0
        plt = plot_average_temperature(model, Tavg)
        show(plt)
        @printf "\n"
    end
end
