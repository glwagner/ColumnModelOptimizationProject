using Oceananigans, Printf, PyPlot

include("utils.jl")

# 
# Model setup
# 

arch = CPU()
#@hascuda arch = GPU() # use GPU if it's available

model = Model(
     arch = arch, 
        N = (64, 64, 32) .* 4, 
        L = (32, 32, 16) .* 4, 
  closure = ConstantIsotropicDiffusivity(ν=5e-5, κ=5e-5),
      eos = LinearEquationOfState(βS=0.),
constants = PlanetaryConstants(f=1e-4)
)

#
# Initial condition, boundary condition, and tracer forcing
#

N² = 1e-8
Fb = 5e-10
Fu = 0.0 #-1e-6
filename(model) = @sprintf("simple_flux_Fb%.1e_Fu%.1e_Lz%d_Nz%d_nu%.0e", 
                           Fb, Fu, model.grid.Lz, model.grid.Nz, model.closure.ν)

# Temperature initial condition
const dTdz = N² / (model.constants.g * model.eos.βT)
const T₀₀, h₀, δh = 20, 5, 2 # deg C

T₀★(z) = T₀₀ + dTdz * (z+h₀+δh) * step(z+h₀, δh)

# Add a bit of surface-concentrated noise to the initial condition
ξ(z) = 1e-1 * rand() * exp(10z/model.grid.Lz) 
T₀(x, y, z) = T₀★(z) + dTdz*model.grid.Lz * ξ(z)

# Sponges
const μ₀ = 1e-1 * (Fb / model.grid.Lz^2)^(1/3)
const δˢ = model.grid.Lz / 10
const zˢ = -9 * model.grid.Lz / 10

"A step function which is 0 above z=0 and 1 below."
@inline step(z, δ) = (1 - tanh(z/δ)) / 2
@inline μ(z) = μ₀ * step(z-zˢ, δˢ) # sponge function

@inline Fuˢ(grid, u, v, w, T, S, i, j, k) = @inbounds -μ(grid.zC[k]) * u[i, j, k] 
@inline Fvˢ(grid, u, v, w, T, S, i, j, k) = @inbounds -μ(grid.zC[k]) * v[i, j, k]
@inline Fwˢ(grid, u, v, w, T, S, i, j, k) = @inbounds -μ(grid.zC[k]) * w[i, j, k]
@inline FTˢ(grid, u, v, w, T, S, i, j, k) = @inbounds  μ(grid.zC[k]) * (T₀★(grid.zC[k]) - T[i, j, k])

# Passive tracer initial condition and forcing
const δ = model.grid.Lz/6
const ν = model.closure.ν
const λ = ν / δ^2
const c₀₀ = 1

c₀(x, y, z) = c₀₀ * (1 + z/model.grid.Lz) #exp(z/δ) 
ν_∂z²_c★(z) = ν/δ^2 * c₀(0, 0, z)

#
# Set passive tracer forcing, boundary conditions and initial conditions
#

#Fc(grid, u, v, w, T, S, i, j, k) = @inbounds -ν_∂z²_c★(grid.zC[k])
#model.forcing = Forcing(Fu=Fuˢ, Fv=Fvˢ, Fw=Fwˢ, FT=FTˢ, FS=Fc)
model.forcing = Forcing(Fu=Fuˢ, Fv=Fvˢ, Fw=Fwˢ, FT=FTˢ) #, FS=Fc)

Fθ = Fb / (model.constants.g * model.eos.βT)
model.boundary_conditions.T.z.top    = BoundaryCondition(Flux, Fθ)
model.boundary_conditions.T.z.bottom = BoundaryCondition(Gradient, dTdz)
model.boundary_conditions.S.z.top    = BoundaryCondition(Value, c₀₀)
model.boundary_conditions.S.z.bottom = BoundaryCondition(Value, 0)
model.boundary_conditions.u.z.top    = BoundaryCondition(Flux, Fu)

set_ic!(model, T=T₀, S=c₀)

#
# Output
#

function savebcs(file, model)
    file["bcs/Fb"] = Fb
    file["bcs/Fu"] = Fu
    file["bcs/dTdz"] = dTdz
    file["bcs/c₀₀"] = c₀₀
    return nothing
end

u(model)  = Array(data(model.velocities.u))
v(model)  = Array(data(model.velocities.v))
w(model)  = Array(data(model.velocities.w))
θ(model)  = Array(data(model.tracers.T))
c(model)  = Array(data(model.tracers.S))

U(model)  = havg(model.velocities.u)
V(model)  = havg(model.velocities.v)
W(model)  = havg(model.velocities.w)
T(model)  = havg(model.tracers.T)
C(model)  = havg(model.tracers.S)
e(model)  = havg(turbulent_kinetic_energy(model))
wT(model) = havg(model.velocities.w * model.tracers.T)

profiles = Dict(:U=>U, :V=>V, :W=>W, :T=>T, :C=>C, :e=>e, :wT=>wT)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ, :c=>c)

profile_writer = JLD2OutputWriter(model, profiles; dir="data", 
                                  prefix=filename(model)*"_profiles", 
                                  init=savebcs, frequency=100, force=true)
                                  
field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, frequency=1000, force=true)

push!(model.output_writers, profile_writer, field_writer)

gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(8, 10), gridspec_kw=gridspec)

ρ₀ = 1035.0
cp = 3993.0

@printf(
    """
    Crunching a (viscous) ocean surface boundary layer with
    
            n : %d, %d, %d
           Fb : %.1e
           Fu : %.1e
            Q : %.2e
          |τ| : %.2e
            ν : %.1e
          1/N : %.1f min
    
    Let's spin the gears.
    
    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, Fb, Fu, 
             -ρ₀*cp*Fb/(model.constants.g*model.eos.βT), abs(ρ₀*Fu), model.closure.ν,
             sqrt(1/N²) / 60
)

# Sensible initial time-step
αν = 1e-2
αu = 5e-1

# spinup
for i = 1:100
    Δt = safe_Δt(model, αu, αν)
    walltime = @elapsed time_step!(model, 1, Δt)
end

# main loop
for i = 1:100
    Δt = safe_Δt(model, αu, αν)
    walltime = @elapsed time_step!(model, 20, Δt)

    makeplot(axs, model)

    @printf("i: %d, t: %.2f hours, cfl: %.2e, wall: %s\n", model.clock.iteration,
            model.clock.time/3600, cfl(Δt, model), prettytime(1e9*walltime))
end
