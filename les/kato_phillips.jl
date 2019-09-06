using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

# From Kato and Phillips (1969).
# Note that [g cm⁻¹ s⁻²] = 10⁻¹ [kg m⁻¹ s⁻²] and [g cm⁻⁴] = 10⁵ [kg m⁻⁴].
τ₀_kato = [0.995, 1.485, 2.12, 2.75] .* 1e-1
ρz_kato = -[1.92, 3.84, 7.69] .* 1e2

prefix = "kato_phillips_long"
Ny = 64
Δt = 1e-3 # initial time-step
τ₀ = τ₀_kato[1]
ρz = ρz_kato[1]

 N = (4Ny, Ny, Ny)
 L = (4 * 0.23, 0.23, 0.23) # meters
 g = 9.81 # m s⁻²
ρ₀ = 1000.0 # kg m⁻³
tf = 240.0 # seconds

const N² = - g * ρz / ρ₀
const Qu = - τ₀ / ρ₀

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.1, Δt=Δt, max_change=1.1, max_Δt=0.1)

# Sponge
@inline μ(z, Lz) = 10.0 * exp(-(z + Lz) / 0.05Lz)
@inline b₀(z) = N² * z

@inline Fu(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]
@inline Fb(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * (Φ.T[i, j, k] - b₀(grid.zC[k]))

# Initial condition
@inline smoothstep(x, x₀, dx) = (1 + tanh((x-x₀) / dx)) / 2
@inline Qu_ramp_up(t) = Qu * smoothstep(t, 1.0, 1.0)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=TimeDependentBoundaryCondition(Flux, Qu_ramp_up))
bbcs = HorizontallyPeriodicBCs(bottom=BoundaryCondition(Gradient, N²))

# Instantiate the model
model = Model(      arch = GPU(),
                       N = N, L = L,
                     eos = LinearEquationOfState(βT=1.0, βS=0.0),
               constants = PlanetaryConstants(f=0.0, g=1.0),
                 closure = AnisotropicMinimumDissipation(ν=1.43e-6, κ=1.2e-9),
                 forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw, FT=Fb),
                     bcs = BoundaryConditions(u=ubcs, T=bbcs)
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
                         MaxAbsFieldDiagnostic(model.diffusivities.νₑ, frequency=frequency),
                         MaxAbsFieldDiagnostic(model.diffusivities.κₑ.T, frequency=frequency),
                         AdvectiveCFL(wizard, frequency=frequency),
                         DiffusiveCFL(wizard, frequency=frequency))

function terse_message(model, walltime, Δt)
    msg1 = @sprintf("i: %d, t: %.2f s, Δt: %.6f s, umax: %.1e ms⁻¹, wmax: %.1e ms⁻¹, ",
                    model.clock.iteration, model.clock.time, Δt, 
                    model.diagnostics[1].data[end], model.diagnostics[2].data[end])

    msg2 = @sprintf("νmax: %.1e m²s⁻¹, κmax: %.1e m²s⁻¹, CFL: %.3f, dCFL: %.3f, wall time: %s\n",
                    model.diagnostics[3].data[end], model.diagnostics[4].data[end],
                    model.diagnostics[5].data[end], model.diagnostics[6].data[end],
                    prettytime(walltime))

    return msg1 * msg2
end

function init_bcs(file, model)
    file["boundary_conditions/top/Qu"] = Qu
    file["boundary_conditions/bottom/N²"] = N²
    return nothing
end

filename = @sprintf("%s_tau%.1f_rhoz%.1f_Ny%d_Nz%d", prefix, τ₀, -ρz, model.grid.Ny, model.grid.Nz)

# Fields
fields = merge(model.velocities, (b=model.tracers.T,),
               (νₑ=model.diffusivities.νₑ, κₑ=model.diffusivities.κₑ.T))
outputs = FieldOutputs(fields)
field_writer = JLD2OutputWriter(model, outputs; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=2GiB, interval=10.0, force=true)

# Averages
avgfluxes = (qb=TimeAveragedFlux(model, :qθ), qu=TimeAveragedFlux(model, :qu))
avgfields = (U=TimeAveragedField(model, model.velocities.u), B=TimeAveragedField(model, model.tracers.T)) 
profile_writer = JLD2OutputWriter(model, merge(avgfluxes, avgfields); dir="data", interval=10.0, 
                                  prefix=filename * "_fluxes", max_filesize=2GiB, force=true)

push!(model.output_writers, field_writer, profile_writer)

function normalize!(a)
    a .-= minimum(a)
    a ./= (maximum(a) - minimum(a))
    return nothing
end

function plot_average_solution(model, U, B, qb)

    canvas = BrailleCanvas
    Lz = model.grid.Lz

    Bnorm = B(model)[2:end-1]
    Unorm = U(model)[2:end-1]

    avg_qb = qb.horizontal_average
    run_diagnostic(model, avg_qb)
    qbnorm = Array(avg_qb.profile)[3:end-1]

    normalize!(Bnorm)
    normalize!(Unorm)
    normalize!(qbnorm)
    
    plt = lineplot(Bnorm, model.grid.zC, name="b",
                    height=20, canvas=canvas, xlim=[0, 1], ylim=[-Lz, 0],
                    xlabel="Normalized u, b, and qᵇ", ylabel="z")

    lineplot!(plt, Unorm, model.grid.zC, name="u")
    lineplot!(plt, qbnorm, model.grid.zF[2:end-1], name="qᵇ")

    return plt
end

U = HorizontalAverage(model, model.velocities.u, frequency=1, return_type=Array)
B = HorizontalAverage(model, model.tracers.T, frequency=1, return_type=Array)
qb = avgfluxes[:qb]

# 
# Run the simulation
#

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)

    walltime = Base.@elapsed time_step!(model, 100, wizard.Δt)

    @printf "%s" terse_message(model, walltime, wizard.Δt)

    if model.clock.iteration % 1000 == 0
        plt = plot_average_solution(model, U, B, qb)
        show(plt)
        @printf "\n"
    end
end
