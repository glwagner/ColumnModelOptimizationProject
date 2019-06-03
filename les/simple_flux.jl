using Distributed
addprocs(1)

@everywhere begin
    using Oceananigans, JLD2, Printf, Distributions, 
          Random, Printf, PyPlot, OceananigansAnalysis,
          Statistics
end

include("utils.jl")
include("cfl_util.jl")
include("jld2_writer.jl")

# Constants
hour = 3600
 day = 24*hour
   g = 9.81
  βT = 2e-4

#
# Initial condition, boundary condition, and tracer forcing
#
      Ny = 64
      Ly = 64

      Nx = 2Ny
      Lx = 2Ly
      Nz = 2Ny
      Lz =  Ly

      Δx = Lx / Nx
      Δz = Lz / Nz

  tfinal = 7*day

      N² = 1e-6
const Fb = 1e-9
const Fu = 0.0 #-1e-4

const T₀₀ = 20.0
const c₀₀ = 1
const kᵘ  = 2π / 4Δx   # wavelength of horizontal divergent surface flux
const aᵘ  = 0.01       # relative amplitude of horizontal divergent surface flux
const dδu = 5Δz        # momentum forcing smoothing scale
const dδθ = 3Δz        # buoyancy forcing smoothing scale
const τˢ = 1000.0      # sponge damping timescale
const δˢ = Lz / 10     # sponge layer width
const zˢ = -Lz + δˢ    # sponge layer central depth

# Buoyancy → temperature
const Fθ   = Fb / (g*βT)
const dTdz = N² / (g * βT)

filename(model) = @sprintf(
                           "simple_flux_Fb%.0e_Fu%.0e_Lz%d_Nz%d",
                           model.attributes.Fb, model.attributes.Fu, 
                           model.grid.Lz, model.grid.Nz
                          )

cbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value, c₀₀),
    bottom = BoundaryCondition(Value, 0.0)
   ))

Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = DefaultBC(), #BoundaryCondition(Flux, Fθ),
    bottom = BoundaryCondition(Gradient, dTdz)
   ))

#
# Sponges, momentum flux, and initial conditions
#

# Vertical noise profile for initial condition
const Lξ = Lz 
Ξ(z) = rand(Normal(0, 1)) * z / Lξ * (1 + z / Lξ)

T₀★(z) = T₀₀ + dTdz * z
T₀(x, y, z) = T₀★(z) + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-4 * Ξ(z)
v₀(x, y, z) = 1e-4 * Ξ(z)
c₀(x, y, z) = 1e-6 * Ξ(z)

"A regularized delta function."
@inline δu(z) = √(π) / (2dδu) * exp(-z^2 / (2dδu^2))
@inline δθ(z) = √(π) / (2dδθ) * exp(-z^2 / (2dδθ^2))

"A step function which is 0 above z=0 and 1 below."
@inline smoothstep(z, δ) = (1 - tanh(z/δ)) / 2

"""
A sponging function that is zero above zˢ, has width δˢ, and 
sponges with timescale τˢ.
"""
@inline sponge(z) = 1/τˢ * smoothstep(z-zˢ, δˢ)

# Momentum forcing: smoothed over surface grid points, plus 
# horizontally-divergence component to stimulate turbulence.
@inline FFu(grid, u, v, w, T, S, i, j, k) = 
    @inbounds -Fu * δu(grid.zC[k]) * (1 + aᵘ * sin(kᵘ * grid.xC[i] + 2π*rand()))

# Relax bottom temperature field to background profile
@inline FTˢ(grid, u, v, w, T, S, i, j, k) = 
    @inbounds -Fθ * δθ(grid.zC[k]) + sponge(grid.zC[k]) * (T₀★(grid.zC[k]) - T[i, j, k])

# 
# Model setup
# 

arch = CPU()
@hascuda arch = GPU() # use GPU if it's available

model = Model(
         arch = arch,
            N = (Nx, Ny, Nz),
            L = (Lx, Ly, Lz), 
      closure = AnisotropicMinimumDissipation(), 
          eos = LinearEquationOfState(βT=βT, βS=0.),
    constants = PlanetaryConstants(f=1e-4, g=g),
      forcing = Forcing(Fu=FFu, FT=FTˢ),
          bcs = BoundaryConditions(T=Tbcs, S=cbcs),
   attributes = (Fb=Fb, Fu=Fu)
)

set_ic!(model, u=u₀, v=v₀, T=T₀, S=c₀)

#
# Output
#

#=
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

U(model)  = Array(havg(model.velocities.u))
V(model)  = Array(havg(model.velocities.v))
W(model)  = Array(havg(model.velocities.w))
T(model)  = Array(havg(model.tracers.T))
C(model)  = Array(havg(model.tracers.S))
#e(model)  = havg(turbulent_kinetic_energy(model))
#wT(model) = havg(model.velocities.w * model.tracers.T)

profiles = Dict(:U=>U, :V=>V, :W=>W, :T=>T, :C=>C) #, :e=>e, :wT=>wT)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ, :c=>c)

profile_writer = JLD2OutputWriter(model, profiles; dir="data", 
                                  prefix=filename(model)*"_profiles", 
                                  init=savebcs, frequency=100, force=true,
                                  asynchronous=true)
                                  
field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, frequency=1000, force=true,
                                asynchronous=true)

push!(model.output_writers, profile_writer, field_writer)
=#

gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(16, 10), gridspec_kw=gridspec)

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
          1/N : %.1f min
           βT : %.2e
     filename : %s
    
    Let's spin the gears.
    
    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, Fb, Fu, 
             -ρ₀*cp*Fb/(model.constants.g*model.eos.βT), abs(ρ₀*Fu),
             sqrt(1/N²) / 60, model.eos.βT, filename(model)
)

function nice_message(model, walltime, Δt) 
    return @sprintf("i: %05d, t: %.4f hours, Δt: %.1f s, wall: %s\n", 
                    model.clock.iteration, model.clock.time/3600, Δt, 
                    prettytime(1e9*walltime))
end

# CFL wizard
wizard = TimeStepWizard(cfl=2e-1, Δt=1.0)

@time time_step!(model, 1, 1e-16) # time first time-step
boundarylayerplot(axs, model)

# Spinup
for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 1, wizard.Δt)
    @printf "%s" nice_message(model, walltime, wizard.Δt)
end

boundarylayerplot(axs, model)

@sync begin
    # Main loop
    while model.clock.time < tfinal
        update_Δt!(wizard, model)
        walltime = @elapsed time_step!(model, 100, wizard.Δt)
        @printf "%s" nice_message(model, walltime, wizard.Δt)
        boundarylayerplot(axs, model)
    end
end
