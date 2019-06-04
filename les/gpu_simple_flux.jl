using Distributed
addprocs(1)

@everywhere begin
    using Oceananigans, OceananigansAnalysis,
            JLD2, Printf, Distributions, Random,
            Printf, Statistics, CuArrays,
            PyPlot
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

      Ny = 128
      Ly = 64

      Nx = 2Ny
      Lx = 2Ly
      Nz = Ny
      Lz = Ly

      Δx = Lx / Nx
      Δz = Lz / Nz

  tfinal = 7*day

      N² = 1e-6
const Fb = 1e-9
const Fu = 0.0 #-1e-4

const T₀₀  = 20.0
const c₀₀  = 1
const kᵘ   = 2π / 4Δx   # wavelength of horizontal divergent surface flux
const aᵘ   = 0.01       # relative amplitude of horizontal divergent surface flux
const dδ = 5Δz          # momentum forcing smoothing scale
const τˢ = 1000.0       # sponge damping timescale
const δˢ = Lz / 10      # sponge layer width
const zˢ = -Lz + δˢ     # sponge layer central depth

# Buoyancy → temperature
      Fθ   = Fb / (g*βT)
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
    top    = BoundaryCondition(Flux, Fθ),
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
@inline δ(z) = √(π) / (2dδ) * exp(-z^2 / (2dδ^2))

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
    @inbounds -Fu * δ(grid.zC[k]) * (1 + aᵘ * sin(kᵘ * grid.xC[i])) # + 2π*rand()))

# Relax bottom temperature field to background profile
@inline FTˢ(grid, u, v, w, T, S, i, j, k) = 
    @inbounds sponge(grid.zC[k]) * (T₀★(grid.zC[k]) - T[i, j, k])

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

u(model)  = Array(parentdata(model.velocities.u))
v(model)  = Array(parentdata(model.velocities.v))
w(model)  = Array(parentdata(model.velocities.w))
θ(model)  = Array(parentdata(model.tracers.T))
c(model)  = Array(parentdata(model.tracers.S))

struct HorizontalAverages{A}
    U :: A
    V :: A
    T :: A
    S :: A
end

function HorizontalAverages(arch::CPU, grid::Grid{FT}) where FT
    U = zeros(FT, 1, 1, grid.Tz)
    V = zeros(FT, 1, 1, grid.Tz)
    T = zeros(FT, 1, 1, grid.Tz)
    S = zeros(FT, 1, 1, grid.Tz)

    HorizontalAverages(U, V, W, T, S)
end

function HorizontalAverages(arch::GPU, grid::Grid{FT}) where FT
    U = CuArray{FT}(undef, 1, 1, grid.Nz)
    V = CuArray{FT}(undef, 1, 1, grid.Nz)
    T = CuArray{FT}(undef, 1, 1, grid.Nz)
    S = CuArray{FT}(undef, 1, 1, grid.Nz)

    HorizontalAverages(U, V, T, S)
end

HorizontalAverages(m::Model{A}) where A = 
    HorizontalAverages(A(), model.grid)

function hmean!(ϕavg, ϕ::Field)
    ϕavg .= mean(parentdata(ϕ), dims=(1, 2))
    return nothing
end

const avgs = HorizontalAverages(model)

function U(model)
    hmean!(avgs.U, model.velocities.u)
    return Array(avgs.U)
end

function V(model)
    hmean!(avgs.V, model.velocities.v)
    return Array(avgs.V)
end

function T(model)
    hmean!(avgs.T, model.tracers.T)
    return Array(avgs.T)
end

function S(model)
    hmean!(avgs.S, model.tracers.S)
    return Array(avgs.S)
end

profiles = Dict(:U=>U, :V=>V, :T=>T, :C=>S)
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
@info "Completed first timestep."

# Spinup
for i = 1:99
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 1, wizard.Δt)
    @printf "%s" nice_message(model, walltime, wizard.Δt)
end

@sync begin
    # Main loop
    while model.clock.time < tfinal
        update_Δt!(wizard, model)
        walltime = @elapsed time_step!(model, 10, wizard.Δt)
        @printf "%s" nice_message(model, walltime, wizard.Δt)
        if model.clock.iteration % 1000 == 0
            @info "Making a plot!"
            @time boundarylayerplot(axs, model)
        end
    end
end

boundarylayerplot(axs, model)
