using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

# From Kato and Phillips (1969).
# Note that [g cm⁻¹ s⁻²] = 10⁻¹ [kg m⁻¹ s⁻²] and [g cm⁻⁴] = 10⁵ [kg m⁻⁴].
τ₀_kato = [0.995, 1.485, 2.12, 2.75] .* 1e-1
ρz_kato = [1.92, 3.84, 7.69] .* 1e2

Ny = 128
N = (2Ny, Ny, Ny)
L = (0.46, 0.23, 0.23) # meters

τ₀ = τ₀_kato[4]
ρz = ρz_kato[1]

ρ₀ = 1000.0 # kg m⁻³
tf = 240.0 # seconds
Δt = 1e-3 # initial time-step

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.5, Δt=Δt, max_change=1.05, max_Δt=0.1)

# Setup bottom sponge layer
α, g, f, θᵣ = 2e-4, 9.81, 0.0, 20.0
@show N² = g * ρz / ρ₀
dθdz = N² / (α*g)

const Qu = -τ₀ / ρ₀
@inline smoothstep(x, x₀, dx) = (1 + tanh((x-x₀) / dx)) / 2
@inline Qu_ramp_up(t) = Qu * smoothstep(t, 2.0, 1.0)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=TimeDependentBoundaryCondition(Flux, Qu_ramp_up))

# Instantiate the model
model = Model(      arch = GPU(),
                       N = N, L = L,
                     eos = LinearEquationOfState(βT=α, βS=0.0),
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

frequency = 1
push!(model.diagnostics, MaxAbsFieldDiagnostic(model.velocities.u, frequency=frequency),
                         MaxAbsFieldDiagnostic(model.velocities.w, frequency=frequency),
                         AdvectiveCFL(wizard, frequency=frequency),
                         DiffusiveCFL(wizard, frequency=frequency))


function init_bcs(file, model)
    file["boundary_conditions/top/Qu"] = Qu
    file["boundary_conditions/bottom/N²"] = N²
    return nothing
end

filename = @sprintf("kato_phillips_Nx%d_Nz%d", model.grid.Nx, model.grid.Nz)

fields = merge(model.velocities, (T=model.tracers.T, νₑ=model.diffusivities.νₑ, κₑ=model.diffusivities.κₑ.T))
outputs = FieldOutputs(fields)
field_writer = JLD2OutputWriter(model, outputs; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=2GiB, interval=10.0, force=true)

average_profiles = Dict{Symbol, Any}()
average_fluxes = Dict(flux=>TimeAveragedFlux(model, flux) for flux in (:wθ, :wu))
average_fields = Dict(:U=>TimeAveragedField(model, model.velocities.u), 
                      :T=>TimeAveragedField(model, model.tracers.T)) 

merge!(average_profiles, average_fluxes, average_fields)
profile_writer = JLD2OutputWriter(model, average_profiles; dir="data", prefix=filename * "fluxes", 
                                  max_filesize=2GiB, interval=1.0, force=true)

push!(model.output_writers, field_writer, profile_writer)

# 
# Run the simulation
#

terse_message(model, walltime, Δt) =
    @sprintf(
    "i: %d, t: %.4f hours, Δt: %.3f s, umax: %.2e ms⁻¹, wmax: %.2e ms⁻¹, adv cfl: %.3f, diff cfl: %.3f, wall time: %s\n",
    model.clock.iteration, model.clock.time/3600, Δt, 
    model.diagnostics[1].data[end], model.diagnostics[2].data[end], 
    model.diagnostics[3].data[end], model.diagnostics[4].data[end],
    prettytime(walltime)
   )

plot_average_temperature(model, Tavg) =
    lineplot(Array(Tavg(model))[2:end-1], model.grid.zC, 
             height=40, canvas=DotCanvas, 
             xlim=[θᵣ-dθdz*model.grid.Lz, θᵣ], ylim=[-model.grid.Lz, 0])

Tavg = HorizontalAverage(model, model.tracers.T, frequency=1)

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = Base.@elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    if model.clock.iteration % 100 == 0
        plt = plot_average_temperature(model, Tavg)
        show(plt)
        @printf "\n"
    end
end
