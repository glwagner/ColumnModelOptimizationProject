using Oceananigans, JLD2, Random, Printf, Statistics, Distributions

macro doesnothavecuda(ex)
    return HAVE_CUDA ? :(nothing) : :($(esc(ex)))
end

@doesnothavecuda include("plot_utils.jl")

@hascuda using CuArrays, CUDAnative, Adapt

include("time_step_wizard.jl")
include("jld2_writer.jl")

# Constants
minute = 60
  hour = 3600
   day = 24*hour
     g = 9.81
    βT = 2e-4

#
# Initial condition, boundary condition, and tracer forcing
#
      FT = Float32
       Δ = 0.5
      Ny = 128
      Ly = Δ * Ny

      Nx = 2Ny
      Lx = 2Ly
      Nz = 2Ny
      Lz = Ly

      Δx = Lx / Nx
      Δz = Lz / Nz

  tfinal = 4*day

# Boundary conditioons and initial condition

cases = Dict(
             1 => (N² = 1e-6, Fb =  1e-8, Fu =  0e-4), # free convection
             2 => (N² = 1e-6, Fb =  1e-9, Fu =  0e-4),
             3 => (N² = 1e-7, Fb =  1e-8, Fu =  0e-4),
             4 => (N² = 1e-7, Fb =  1e-9, Fu =  0e-4),
             5 => (N² = 1e-6, Fb =  0e-8, Fu = -1e-4), # neutral wind
             6 => (N² = 2e-5, Fb =  5e-9, Fu = -1e-4), # unstable wind
             7 => (N² = 1e-6, Fb =  5e-9, Fu = -1e-4), # unstable wind
             8 => (N² = 1e-6, Fb = -1e-9, Fu =  0e-4)  # stable wind
            )

# 
#       N²   |    Fb    |   Fu
#    -------------------------
# 1.   1e-6  |   1e-8   |  -0e-4    # free convection
# 2.   1e-6  |   1e-9   |  -0e-4   
# 3.   1e-7  |   1e-8   |  -0e-4   
# 4.   1e-7  |   1e-9   |  -0e-4   
# 5.   1e-6  |    0     |  -1e-4    # neutral wind
# 6.   1e-6  |   1e-8   |  -1e-4    # unstable wind
# 7.   1e-6  |  -1e-9   |  -1e-4    # stable wind

case = 7
_Fu = cases[case].Fu
_Fb = cases[case].Fb

const N² = FT(cases[case].N²)
const Fb = FT(_Fu)
const Fu = FT(_Fb)

# Constant parameters: temperature/tracer bcs, numerical parameters
const T₀₀ = FT( 20.0     ) 
const S₀₀ = FT( 1        )
const δh  = FT( 2Δz      )  # momentum forcing smoothing height
const kᵘ  = FT( 2π / 4Δx )  # wavelength of horizontal divergent surface flux
const aᵘ  = FT( 0.01     )  # relative amplitude of horizontal divergent surface flux

# Buoyancy → temperature
const Fθ   = FT( Fb / (g*βT) ) 
const dTdz = FT( N² / (g*βT) )

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
    top    = DefaultBC(),
    bottom = BoundaryCondition(Gradient, FT(dTdz))
   ))

#
# Sponges, momentum flux, and initial conditions
#

# Vertical noise profile for initial condition
Ξ(z) = rand(Normal(0, 1)) * z / Lz * (1 + z / Lz)

T₀★(z) = T₀₀ + dTdz * z
S₀★(z) = S₀₀ * (1 + z/Lz)

T₀(x, y, z) = T₀★(z) + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-4 * Ξ(z)
v₀(x, y, z) = 1e-4 * Ξ(z)
S₀(x, y, z) = S₀★(z)

"A regularized delta function."
@inline δ(z) = sqrt(π) / (2δh) * exp(-z^2 / (2δh^2))

# Momentum forcing: smoothed over surface grid points, plus 
# horizontally-divergence component to stimulate turbulence.
@inline T_forcing(grid, u, v, w, T, S, i, j, k, iter) = @inbounds -Fθ * δ(grid.zC[k])
@inline u_forcing(grid, u, v, w, T, S, i, j, k, iter) = 
@inbounds -Fu * δ(grid.zC[k]) * (1 + aᵘ * CUDAnative.sin(kᵘ * grid.xF[i] + iter*2π))

#forcing = Forcing(FT=T_forcing)
forcing = Forcing(Fu=u_forcing, FT=T_forcing)

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
      forcing = forcing,
          bcs = BoundaryConditions(T=Tbcs, S=Sbcs),
   attributes = (Fb=Fb, Fu=Fu)
)

set_ic!(model, u=u₀, v=v₀, T=T₀, S=S₀)

#
# Output
#

function savebcs(file, model)
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fb"] = Fb
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/top/S₀"] = S₀₀
    file["boundary_conditions/bottom/dTdz"] = dTdz
    file["boundary_conditions/bottom/dbdz"] = dTdz * g * βT
      file["boundary_conditions/Bz"] = 
     file["boundary_conditions/S₀₀"] = S₀₀
    return nothing
end

u(model) = Array{Float32}(model.velocities.u.data.parent)
v(model) = Array{Float32}(model.velocities.v.data.parent)
w(model) = Array{Float32}(model.velocities.w.data.parent)
θ(model) = Array{Float32}(model.tracers.T.data.parent)
s(model) = Array{Float32}(model.tracers.S.data.parent)
ν(model) = Array{Float32}(model.diffusivities.νₑ.parent)

function hmean!(ϕavg, ϕ::Field)
    ϕavg .= mean(ϕ.data.parent, dims=(1, 2))
    return nothing
end

const avgs = HorizontalAverages(model)
const vplanes = VerticalPlanes(model)

function U(model)
    hmean!(avgs.U, model.velocities.u)
    return Array{Float32}(avgs.U)
end

function V(model)
    hmean!(avgs.V, model.velocities.v)
    return Array{Float32}(avgs.V)
end

function T(model)
    hmean!(avgs.T, model.tracers.T)
    return Array{Float32}(avgs.T)
end

function S(model)
    hmean!(avgs.S, model.tracers.S)
    return Array{Float32}(avgs.S)
end

function uxz(model) = Array{Float32}(view(model.velocities.u.parent, :, 1, :)
function vxz(model) = Array{Float32}(view(model.velocities.v.parent, :, 1, :)
function wxz(model) = Array{Float32}(view(model.velocities.w.parent, :, 1, :)
function Txz(model) = Array{Float32}(view(model.tracers.T.parent, :, 1, :)
function Sxz(model) = Array{Float32}(view(model.tracers.S.parent, :, 1, :)

profiles = Dict(:U=>U, :V=>V, :T=>T, :S=>S)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ, :s=>s, :ν=>ν)
  planes = Dict(:u=>uxz, :v=>vxz, :w=>wxz, :θ=>Txz, :s=>Sxz)

profile_writer = JLD2OutputWriter(model, profiles; dir="data", 
                                  prefix=filename(model)*"_profiles", 
                                  init=savebcs, interval=0.5*hour, force=true)

plane_writer = JLD2OutputWriter(model, planes; dir="data", 
                                prefix=filename(model)*"_planes", 
                                  init=savebcs, interval=5minute, force=true)
                                  
field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, interval=2hour, force=true)

push!(model.output_writers, plane_writer, profile_writer, field_writer)

ρ₀ = 1035.0
cp = 3993.0

@printf(
    """
    Crunching an ocean surface boundary layer with
    
            n : %d, %d, %d
            L : %d, %d, %d m
            Δ : %.2f, %.2f, %.2f m
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

@doesnothavecuda begin
    gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
    fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(8, 10), gridspec_kw=gridspec)
    boundarylayerplot(axs, model)
end

# CFL wizard
wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.1, max_Δt=90.0)

# Spinup 
for i = 1:1000
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, FT(wizard.Δt))
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

@doesnothavecuda boundarylayerplot(axs, model)

ifig = 1

# Main loop
while model.clock.time < tfinal
    global ifig

    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 50, FT(wizard.Δt))

    @printf "%s" terse_message(model, walltime, wizard.Δt)

    @doesnothavecuda boundarylayerplot(axs, model)
    @doesnothavecuda savefig(joinpath("plots", filename(model) * "_$ifig.png"), dpi=480)
    ifig += 1
end
