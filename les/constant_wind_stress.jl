using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

include("utils.jl")

# 
# Model set-up
#

Nx = Nz = 128
Lx = Lz = 64 

# From Van Roekel et al (JAMES, 2018)
N² = 9.81e-5 
Qu = -9.66e-5
 f = 0.0
tf = 1day
Δt = 0.1 # initial time-step

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.5, Δt=Δt, max_change=1.1, max_Δt=10.0)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=BoundaryCondition(Flux, Qu))

# Setup bottom sponge layer
αθ, g = 2e-4, 9.81
const dθdz = N² / (g*αθ)
const θᵣ = 20.0
const Δμ = 3.2

@inline μ(z, Lz) = 0.02 * exp(-(z+Lz) / Δμ)
@inline θ₀(z) = θᵣ + dθdz * z
@inline Fu(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]
@inline Fθ(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * (Φ.T[i, j, k] - θ₀(grid.zC[k]))

# Instantiate the model
model = Model(      arch = GPU(),
                       N = (Nx, Nx, Nz), L = (Lx, Lx, Lz),
                     eos = LinearEquationOfState(βT=αθ, βS=0.0),
               constants = PlanetaryConstants(f=f, g=g),
                 closure = AnisotropicMinimumDissipation(Cb=1.0),
                 forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw, FT=Fθ),
                     bcs = BoundaryConditions(u=ubcs)
)

# Set initial condition
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
uᵢ(x, y, z) = 1e-3 * Ξ(z)
θᵢ(x, y, z) = θᵣ + dθdz * z + 1e-3 * dθdz * model.grid.Lz * Ξ(z)
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, T=θᵢ)

#
# Set up time stepping, output and diagnostics
#

function init_bcs(file, model)
    file["boundary_conditions/top/Qb"] = 0.0
    file["boundary_conditions/top/Qθ"] = 0.0
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
filename = @sprintf("wind_stress_Nx%d_Nz%d_%s_bmod", Nx, Nz, closurename(model.closure))

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
