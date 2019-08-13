using Oceananigans, Random, Printf

# 
# Model set-up
#

# Simulation parameters
  DT = Float64                  # Data type
   N = 256                      # Resolution    
   Δ = 1.0                      # Grid spacing
  tf = 8day                     # Final simulation time

# Physical constants
  βT = 2e-4                     # Thermal expansion coefficient
   g = 9.81                     # Gravitational acceleration
  Fθ = 1e-8 / (g*βT)            # Temperature flux
dTdz = 1e-4 / (g*βT)            # Initial temperature gradient

# Create boundary conditions
ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, 1e-4))
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Fθ),
                                bottom = BoundaryCondition(Gradient, dTdz))

# Instantiate the model
model = Model(float_type = DT, 
                    arch = HAVE_CUDA ? GPU() : CPU(),
                       N = (N, N, 1024),
                       L = (N*Δ, N*Δ, N*Δ),
                     eos = LinearEquationOfState(DT, βT=βT, βS=0.0),
               constants = PlanetaryConstants(DT, f=1e-4, g=g),
                 closure = AnisotropicMinimumDissipation(DT), # closure = ConstantSmagorinsky(DT),
                     bcs = BoundaryConditions(u=ubcs, T=Tbcs))

#=
# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
T₀(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-1 * Ξ(z)
w₀(x, y, z) = 1e-3 * Ξ(z)
S₀(x, y, z) = 1e0 * Ξ(z)

set_ic!(model, u=u₀, w=w₀, T=T₀, S=S₀)
=#

# 
# Run the simulation
#

function terse_message(model, walltime, Δt)
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)

    return @sprintf(
        "i: %d, t: %.4f hours, Δt: %.1f s, wmax: %.6f ms⁻¹, cfl: %.3f, wall time: %s\n",
        model.clock.iteration, model.clock.time/3600, Δt, wmax, cfl, prettytime(1e9*walltime),
       )
end

# Spin up
wizard = TimeStepWizard(cfl=0.01, Δt=1.0, max_change=1.1, max_Δt=90.0)

for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

# Reset CFL condition values
wizard.cfl = 0.2
wizard.max_change = 1.5

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 100, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end
