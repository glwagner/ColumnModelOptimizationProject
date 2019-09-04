using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

τ₀_table = [0.995, 1.485, 2.12, 2.75] .* 1e-5
dρdz_table = [1.92, 3.84, 7.69] .* 1e2

# From Kato and Phillips (1969)
Ny = 32 
N = (2Ny, Ny, Ny)
L = (0.46, 0.23, 0.23)

τ₀ = τ₀_table[4]
dρdz = dρdz_table[3]

ρ₀ = 1000.0
tf = 240.0
Δt = 1e-2 # initial time-step

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.5, Δt=Δt, max_change=1.05, max_Δt=1.0)

# Setup bottom sponge layer
αθ, g, f, θᵣ = 2e-4, 9.81, 0.0, 20.0
N² = g * dρdz / ρ₀
dθdz = N² / (g*αθ)

const Qu = -τ₀ / ρ₀
@inline smoothstep(x, x₀, dx) = (1 + tanh((x-x₀) / dx)) / 2
@inline Qu_ramp_up(t) = Qu * smoothstep(t, 2.0, 1.0)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=TimeDependentBoundaryCondition(Flux, Qu_ramp_up))

# Instantiate the model
model = Model(      arch = GPU(),
                       N = N, L = L,
                     eos = LinearEquationOfState(βT=αθ, βS=0.0),
               constants = PlanetaryConstants(f=f, g=g),
                 closure = AnisotropicMinimumDissipation(Cb=1.0, κ=1.2e-9),
                     bcs = BoundaryConditions(u=ubcs)
)

# Set initial condition
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
uᵢ(x, y, z) = 1e-3 * sqrt(abs(Qu)) * Ξ(z)
θᵢ(x, y, z) = θᵣ + dθdz * z + 1e-3 * dθdz * model.grid.Lz * Ξ(z)
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, T=θᵢ)

#
# Set up time stepping, output and diagnostics
#

function init_bcs(file, model)
    file["boundary_conditions/top/Qu"] = Qu
    file["boundary_conditions/bottom/N²"] = N²
    return nothing
end

function p(model)
    model.pressures.pNHS.data.parent .+= model.pressures.pHY′.data.parent
    return Array(model.pressures.pNHS.data.parent)
end

fields = Dict{Symbol, Any}()
merge!(fields, FieldOutputs(model.velocities))
merge!(fields, FieldOutputs((T=model.tracers.T, νₑ=model.diffusivities.νₑ)))
fields[:p] = p

if typeof(model.closure) <: AnisotropicMinimumDissipation
    fields[:κₑ] = FieldOutput(model.diffusivities.κₑ.T)
end

closurename(::AnisotropicMinimumDissipation) = "amd"
closurename(::ConstantSmagorinsky) = "smag"
filename = @sprintf("kato_phillips_Nx%d_Nz%d_%s", model.grid.Nx, model.grid.Nz, closurename(model.closure))

field_writer = JLD2OutputWriter(model, fields; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=2GiB, interval=3hour, force=true)
push!(model.output_writers, field_writer)

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

plot_average_temperature(model, Tavg) =
    lineplot(Array(Tavg(model))[2:end-1], model.grid.zC, 
             height=40, canvas=DotCanvas, xlim=[20-dθdz*Lz, 20], ylim=[-Lz, 0])

Tavg = HorizontalAverage(model, model.tracers.T, frequency=1)

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
