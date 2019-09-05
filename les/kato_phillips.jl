using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

# From Kato and Phillips (1969).
# Note that [g cm⁻¹ s⁻²] = 10⁻¹ [kg m⁻¹ s⁻²] and [g cm⁻⁴] = 10⁵ [kg m⁻⁴].
τ₀_kato = [0.995, 1.485, 2.12, 2.75] .* 1e-1
ρz_kato = -[1.92, 3.84, 7.69] .* 1e2

Ny = 256
Δt = 1e-4 # initial time-step
τ₀ = τ₀_kato[1]
ρz = ρz_kato[1]

 N = (2Ny, Ny, Ny)
 L = (0.46, 0.23, 0.23) # meters
 g = 9.81 # m s⁻²
ρ₀ = 1000.0 # kg m⁻³
tf = 240.0 # seconds
@show N² = - g * ρz / ρ₀

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.5, Δt=Δt, max_change=1.05, max_Δt=0.1)

const Qu = -τ₀ / ρ₀
@inline smoothstep(x, x₀, dx) = (1 + tanh((x-x₀) / dx)) / 2
@inline Qu_ramp_up(t) = Qu * smoothstep(t, 2.0, 1.0)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=TimeDependentBoundaryCondition(Flux, Qu_ramp_up))

# Instantiate the model
model = Model(      arch = GPU(),
                       N = N, L = L,
                     eos = LinearEquationOfState(βT=1.0, βS=0.0),
               constants = PlanetaryConstants(f=0.0, g=1.0),
                 closure = AnisotropicMinimumDissipation(ν=1.43e-6, κ=1.2e-9),
                     bcs = BoundaryConditions(u=ubcs)
)

# Set initial condition
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
uᵢ(x, y, z) = 1e-3 * sqrt(abs(Qu)) * Ξ(z)
bᵢ(x, y, z) = N² * z + 1e-3 * N² * model.grid.Lz * Ξ(z)
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, T=bᵢ)

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

filename = @sprintf("kato_phillips_τ%.1f_ρ%.1f_Nx%d_Nz%d", τ₀, -ρz, model.grid.Nx, model.grid.Nz)

fields = merge(model.velocities, (b=model.tracers.T,),
               (νₑ=model.diffusivities.νₑ, κₑ=model.diffusivities.κₑ.T))
outputs = FieldOutputs(fields)
field_writer = JLD2OutputWriter(model, outputs; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=2GiB, interval=10.0, force=true)

average_profiles = Dict{Symbol, Any}()
average_fluxes = Dict(:wb=>TimeAveragedFlux(model, :wθ), :wu=>TimeAveragedFlux(model, :wu))
average_fields = Dict(:U=>TimeAveragedField(model, model.velocities.u), 
                      :B=>TimeAveragedField(model, model.tracers.T)) 

merge!(average_profiles, average_fluxes, average_fields)
profile_writer = JLD2OutputWriter(model, average_profiles; dir="data", prefix=filename * "_fluxes", 
                                  max_filesize=2GiB, interval=0.2, force=true)

push!(model.output_writers, field_writer, profile_writer)

# 
# Run the simulation
#

terse_message(model, walltime, Δt) =
    @sprintf(
    "i: %d, t: %.2f s, Δt: %.4f s, umax: %.1e ms⁻¹, wmax: %.1e ms⁻¹, CFL: %.3f, dCFL: %.3f, wall time: %s\n",
    model.clock.iteration, model.clock.time, Δt, 
    model.diagnostics[1].data[end], model.diagnostics[2].data[end], 
    model.diagnostics[3].data[end], model.diagnostics[4].data[end],
    prettytime(walltime)
   )

function normalize!(a)
    a .-= minimum(a)
    a ./= (maximum(a) - minimum(a))
    return nothing
end

function plot_average_solution(model, U, B)

    canvas = BrailleCanvas
    Lz = model.grid.Lz

    Bnorm = B(model)[2:end-1]
    Unorm = U(model)[2:end-1]

    normalize!(Bnorm)
    normalize!(Unorm)
    
    plt = lineplot(Bnorm, model.grid.zC, name="buoyancy",
                    height=20, canvas=canvas, xlim=[0, 1], ylim=[-Lz, 0],
                    xlabel="Normalized buoyancy and velocity", ylabel="z")

    lineplot!(plt, Unorm, model.grid.zC, name="x-velocity")

    return plt
end

U = HorizontalAverage(model, model.velocities.u, frequency=1, return_type=Array)
B = HorizontalAverage(model, model.tracers.T, frequency=1, return_type=Array)

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = Base.@elapsed time_step!(model, 100, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    if model.clock.iteration % 1000 == 0
        plt = plot_average_solution(model, U, B)
        show(plt)
        @printf "\n"
    end
end
