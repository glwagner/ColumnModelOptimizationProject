using Oceananigans, JLD2, Random, Printf, Statistics, Distributions

macro doesnothavecuda(ex)
    return HAVE_CUDA ? :(nothing) : :($(esc(ex)))
end

@doesnothavecuda include("plot_utils.jl")

# Constants
 g = 9.81
βT = 2e-4

#
# Initial condition, boundary condition, and tracer forcing
#

TFL = Float64
  Δ = 0.5
 Ny = 16
 Ly = Δ * Ny
 Fu = -1e-6
 Fb = 0.0
 N² = 1e-6

tfinal = 8day

arch = CPU()
@hascuda arch = GPU() # use GPU if it's available

# Buoyancy → temperature
Fθ   = Fb / (g*βT)
dTdz = N² / (g*βT)
 T₀₀ = 20.0

filename(model) = @sprintf(
                           "simple_flux_Fb%.0e_Fu%.0e_Nsq%.0e_Lz%d_Nz%d",
                           model.attributes.Fb, model.attributes.Fu, N²,
                           model.grid.Lz, model.grid.Nz
                          )

# 
# Model setup
# 

Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, TFL(Fθ)),
    bottom = BoundaryCondition(Gradient, TFL(dTdz))
   ))

ubcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, TFL(Fu))
   ))

model = Model(
   float_type = TFL,
         arch = arch,
            N = (2Ny, Ny, 4Ny),
            L = (2Ly, Ly,  Ly), 
      closure = AnisotropicMinimumDissipation(TFL), 
          eos = LinearEquationOfState(TFL, βT=βT),
    constants = PlanetaryConstants(TFL, f=1e-4, g=g),
      forcing = Forcing(),
          bcs = BoundaryConditions(T=Tbcs, u=ubcs),
   attributes = (Fb=Fb, Fu=Fu, dTdz=dTdz, T₀₀=T₀₀)
)

#
# Initial conditions
#

# Vertical noise profile for initial condition
Ξ(z) = rand(Normal(0, 1)) * z / model.grid.Lz * (1 + z / model.grid.Lz)

T₀(x, y, z) = T₀₀ + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-4 * Ξ(z)
v₀(x, y, z) = 1e-4 * Ξ(z)


set_ic!(model, u=u₀, v=v₀, T=T₀)

#
# Output
#

function savebcs(file, model)
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fb"] = Fb
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/bottom/dTdz"] = dTdz
    file["boundary_conditions/bottom/dbdz"] = dTdz * g * βT
    return nothing
end

u(model) = Array{Float32}(model.velocities.u.data.parent)
v(model) = Array{Float32}(model.velocities.v.data.parent)
w(model) = Array{Float32}(model.velocities.w.data.parent)
θ(model) = Array{Float32}(model.tracers.T.data.parent)
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

uxz(model) = Array{Float32}(model.velocities.u.data.parent[:, 3, :])
vxz(model) = Array{Float32}(model.velocities.v.data.parent[:, 3, :])
wxz(model) = Array{Float32}(model.velocities.w.data.parent[:, 3, :])
Txz(model) = Array{Float32}(model.tracers.T.data.parent[:, 3, :])

profiles = Dict(:U=>U, :V=>V, :T=>T)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ)
  planes = Dict(:uxz=>uxz, :vxz=>vxz, :wxz=>wxz, :θxz=>Txz)

profile_writer = JLD2OutputWriter(model, profiles; dir="data", 
                                  prefix=filename(model)*"_profiles", 
                                  init=savebcs, interval=0.5*hour, force=true)

plane_writer = JLD2OutputWriter(model, planes; dir="data", 
                                prefix=filename(model)*"_planes", 
                                init=savebcs, interval=2minute, force=true)
                                  
field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, interval=2hour, force=true)

#push!(model.output_writers, plane_writer, profile_writer, field_writer)
#push!(model.output_writers, profile_writer, field_writer)

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

function terse_message(model, walltime, Δt) 
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)

    return @sprintf(
        "i: %d, t: %.4f hours, Δt: %.1f s, wmax: %.6f ms⁻¹, cfl: %.3f, wall time: %s\n",
        model.clock.iteration, model.clock.time/3600, Δt, wmax, cfl, prettytime(1e9*walltime),
       ) 
end

@doesnothavecuda begin
    gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
    fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(8, 10), gridspec_kw=gridspec)
    boundarylayerplot(axs, model)
end

# CFL wizard
wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.1, max_Δt=90.0)

# Spinup 
for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, TFL(wizard.Δt))
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

@doesnothavecuda boundarylayerplot(axs, model)

# Reset CFL condition values
wizard.cfl = 0.2
wizard.max_change = 1.5
ifig = 1

# Main loop
while model.clock.time < tfinal
    global ifig

    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 100, TFL(wizard.Δt))

    @printf "%s" terse_message(model, walltime, wizard.Δt)

    @doesnothavecuda boundarylayerplot(axs, model)
    ifig += 1
end
