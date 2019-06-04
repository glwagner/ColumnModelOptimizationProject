using Distributed
addprocs(1)

@everywhere begin
    using Oceananigans, JLD2, Printf, Distributions, 
          Random, Printf, OceananigansAnalysis,
          Statistics
end

macro doesnothavecuda(ex)
    return HAVE_CUDA ? :(nothing) : :($(esc(ex)))
end

@hascuda @everywhere using CuArrays, CUDAnative, CuArrays.CURAND

#@hascuda CURAND.seed!()

@doesnothavecuda include("plot_utils.jl")
include("time_step_wizard.jl")
include("jld2_writer.jl")

# Constants
hour = 3600
 day = 24*hour
   g = 9.81
  βT = 2e-4

#
# Initial condition, boundary condition, and tracer forcing
#
      FT = Float64
       Δ = 1.0
      Ny = 128
      Ly = Δ * Ny

      Nx = 2Ny
      Lx = 2Ly
      Nz = 2Ny
      Lz =  Ly

      Δx = Lx / Nx
      Δz = Lz / Nz

  tfinal = 7*day

# Boundary conditioons and initial condition
      N²  = FT( 1e-7 ) 
const Fb  = FT( 1e-9 )
const Fu  = FT( 0.0  ) #-1e-4
const T₀₀ = FT( 20.0 ) 
const S₀₀ = FT( 1    )

# Surface momentum forcing
const kᵘ  = FT( 2π / 4Δx )  # wavelength of horizontal divergent surface flux
const aᵘ  = FT( 0.01     )  # relative amplitude of horizontal divergent surface flux

# Sponges
const hδu = FT( 5Δz      )  # momentum forcing smoothing height
const τˢ  = FT( 1000.0   )  # sponge damping timescale
const δˢ  = FT( Lz / 20  )  # sponge layer width
const zˢ  = FT( -Lz + δˢ )  # sponge layer central depth

# Buoyancy → temperature
const Fθ   = FT( Fb / (g*βT)   ) 
const dTdz = FT( N² / (g * βT) )

filename(model) = @sprintf(
                           "simple_flux_Fb%.0e_Fu%.0e_Nsq%.0e_Lz%d_Nz%d",
                           model.attributes.Fb, model.attributes.Fu, N²,
                           model.grid.Lz, model.grid.Nz
                          )

Sbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value, FT(S₀₀)),
    bottom = BoundaryCondition(Value, -zero(FT))
   ))

Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, Fθ),
    bottom = BoundaryCondition(Gradient, FT(dTdz))
   ))

#
# Sponges, momentum flux, and initial conditions
#

# Vertical noise profile for initial condition
const Lξ = Lz 
Ξ(z) = rand(Normal(0, 1)) * z / Lξ * (1 + z / Lξ)

T₀★(z) = T₀₀ + dTdz * z
S₀★(z) = S₀₀ * (1 + z/Lξ)
T₀(x, y, z) = T₀★(z) + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-4 * Ξ(z)
v₀(x, y, z) = 1e-4 * Ξ(z)
S₀(x, y, z) = S₀★(z)

"A regularized delta function."
@inline δu(z) = sqrt(π) / (2hδu) * exp(-z^2 / (2hδu^2))

"A step function which is 0 above z=0 and 1 below."
@inline smoothstep(z, δ) = (1 - tanh(z/δ)) / 2

"""
A sponging function that is zero above zˢ, has width δˢ, and 
sponges with timescale τˢ.
"""
@inline sponge(z) = 1/τˢ * smoothstep(z-zˢ, δˢ)

# Momentum forcing: smoothed over surface grid points, plus 
# horizontally-divergence component to stimulate turbulence.
@doesnothavecuda @inline FFu(grid, u, v, w, T, S, i, j, k) = 
    @inbounds -Fu * δu(grid.zC[k]) * (1 + aᵘ * sin(kᵘ * grid.xC[i] + 2π*rand()))

@hascuda @inline function FFu(grid, u, v, w, T, S, i, j, k)
    ξ = CuArrays.rand() 
    return @inbounds -Fu * δu(grid.zC[k]) * (1 + aᵘ * CUDAnative.sin(kᵘ * grid.xC[i] + 2π*ξ))
end

# 
# Model setup
# 

arch = CPU()
@hascuda arch = GPU() # use GPU if it's available

model = Model(
   float_type = FT,
         arch = arch,
            N = (Nx, Ny, Nz),
            L = (Lx, Ly, Lz), 
      closure = AnisotropicMinimumDissipation(FT), 
          eos = LinearEquationOfState(FT, βT=βT, βS=0.),
    constants = PlanetaryConstants(FT, f=1e-4, g=g),
      forcing = Forcing(Fu=FFu),
          bcs = BoundaryConditions(T=Tbcs, S=Sbcs),
   attributes = (Fb=Fb, Fu=Fu)
)

set_ic!(model, u=u₀, v=v₀, T=T₀, S=S₀)

#
# Output
#

function savebcs(file, model)
    file["boundary_conditions/Fb"] = Fb
    file["boundary_conditions/Fu"] = Fu
    file["boundary_conditions/dTdz"] = dTdz
    file["boundary_conditions/Bz"] = dTdz * g * βT
    file["boundary_conditions/S₀₀"] = S₀₀
    return nothing
end

u(model) = Array(model.velocities.u.data.parent)
v(model) = Array(model.velocities.v.data.parent)
w(model) = Array(model.velocities.w.data.parent)
θ(model) = Array(model.tracers.T.data.parent)
s(model) = Array(model.tracers.S.data.parent)

function hmean!(ϕavg, ϕ::Field)
    ϕavg .= mean(ϕ.data.parent, dims=(1, 2))
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

profiles = Dict(:U=>U, :V=>V, :T=>T, :S=>S)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ, :s=>s)

profile_writer = JLD2OutputWriter(model, profiles; dir="data", 
                                  prefix=filename(model)*"_profiles", 
                                  init=savebcs, frequency=100, force=true)
                                  
field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, frequency=200, force=true)

push!(model.output_writers, profile_writer, field_writer)

ρ₀ = 1035.0
cp = 3993.0

@printf(
    """
    Crunching a (viscous) ocean surface boundary layer with
    
            n : %d, %d, %d
            L : %d, %d, %d m
            Δ : %.1f, %.1f, %.1f m
           Fb : %.1e m² s⁻³
           Fu : %.1e m² s⁻²
            Q : %.2e W m⁻²
          |τ| : %.2e kg m⁻¹ s⁻²
           N² : %.1e s⁻²
           βT : %.2e (⁰C)⁻¹
     filename : %s
    
    Let's spin.
    
    """, 
    model.grid.Nx, model.grid.Ny, model.grid.Nz, 
    model.grid.Lx, model.grid.Ly, model.grid.Lz, 
    model.grid.Δx, model.grid.Δy, model.grid.Δz, 
    Fb, Fu, -ρ₀*cp*Fb/(model.constants.g*model.eos.βT), abs(ρ₀*Fu),
    N², model.eos.βT, filename(model)
)

function nice_message(model, walltime, Δt) 

    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / cell_advection_timescale(model)

    return @sprintf(
        "i: %05d, t: %.4f hours, Δt: %.1f s, cfl: %.3f, max w: %.6f m s⁻¹, wall: %s\n", 
                    model.clock.iteration, model.clock.time/3600, Δt, 
                    cfl, wmax, prettytime(1e9*walltime))
end

# CFL wizard
wizard = TimeStepWizard(cfl=0.05, Δt=1.0, max_change=1.1, max_Δt=90.0)
@info "Completed first timestep."

@time time_step!(model, 1, 1e-16) # time first time-step

@doesnothavecuda begin
    gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
    fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(8, 10), gridspec_kw=gridspec)
    boundarylayerplot(axs, model)
end

# Spinup
for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, FT(wizard.Δt))
    @printf "%s" nice_message(model, walltime, wizard.Δt)
end

@doesnothavecuda boundarylayerplot(axs, model)
wizard.cfl = 0.2
wizard.max_change = 1.5
ifig = 1

@sync begin
    # Main loop
    while model.clock.time < tfinal
        global ifig

        update_Δt!(wizard, model)
        walltime = @elapsed time_step!(model, 100, FT(wizard.Δt))

        @printf "%s" nice_message(model, walltime, wizard.Δt)

        @doesnothavecuda boundarylayerplot(axs, model)
        @doesnothavecuda savefig(joinpath("plots", filename(model) * "_$ifig.png"), dpi=480)
        ifig += 1
    end
end
