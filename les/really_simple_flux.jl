using Oceananigans, Random, Distributions

   N = 32
   Δ = 1.0
  Fθ = 1e-4
dTdz = 0.01

# Instantiate a model
Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, Fθ),
    bottom = BoundaryCondition(Gradient, dTdz) ))

model = Model(    arch = CPU(), # GPU(),
                     N = (N, N, 2N),
                     L = (N*Δ, N*Δ, N*Δ),
               closure = AnisotropicMinimumDissipation(),
                   bcs = BoundaryConditions(T=Tbcs))

# Set initial condition
Ξ(z) = rand(Normal(0, 1)) * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
T₀(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)

set_ic!(model, T=T₀)

# Run the model (1000 steps with Δt = 1.0)
time_step!(model, 1000, 1.0)
