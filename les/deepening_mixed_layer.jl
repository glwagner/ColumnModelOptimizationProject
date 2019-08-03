using Oceananigans, Random, Printf

using GPUifyLoops: @launch, @loop
using Oceananigans: launch_config, device
using Oceananigans.TurbulenceClosures: ▶x_caa, ▶y_aca, ▶z_aac

# Set `true` to use PyPlot to show the evolution of vertical velocity
makeplot = true

# 
# Model set-up
#

# Two cases from Van Roekel et al (JAMES, 2018)
parameters = Dict(
    :free_convection => Dict(:Fb=>3.39e-8, :Fu=>0.0,     :f=>1e-4, :N²=>1.96e-5),
    :wind_stress     => Dict(:Fb=>0.0,     :Fu=>9.66e-5, :f=>0.0,  :N²=>9.81e-5)
   )

# Simulation parameters
case = :wind_stress
  DT = Float64                  # Data type
   N = 128                      # Resolution    
   Δ = 0.5                      # Grid spacing
  tf = 8day                     # Final simulation time
  N² = parameters[case][:N²]
  Fb = parameters[case][:Fb]
  Fu = parameters[case][:Fu]
   f = parameters[case][:f]

# Physical constants
  βT = 2e-4                     # Thermal expansion coefficient
   g = 9.81                     # Gravitational acceleration
  Fθ = Fb / (g*βT)              # Temperature flux
dTdz = N² / (g*βT)              # Initial temperature gradient

# Create boundary conditions
ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Fu))
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Fθ),
                                bottom = BoundaryCondition(Gradient, dTdz))

# Instantiate the model
model = Model(float_type = DT, 
                    arch = HAVE_CUDA ? GPU() : CPU(),
                       N = (N, 4, 2N),
                       L = (N*Δ, N*Δ, N*Δ),
                     eos = LinearEquationOfState(DT, βT=βT, βS=0.0),
               constants = PlanetaryConstants(DT, f=f, g=g),
                 closure = AnisotropicMinimumDissipation(DT), # closure = ConstantSmagorinsky(DT),
                     bcs = BoundaryConditions(u=ubcs, T=Tbcs))

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
T₀(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
u₀(x, y, z) = 1e-1 * Ξ(z)
w₀(x, y, z) = 1e-3 * Ξ(z)
S₀(x, y, z) = 1e0 * Ξ(z)

set_ic!(model, u=u₀, w=w₀, T=T₀, S=S₀)

#
# Set up output
#

function init_savebcs(file, model)
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fb"] = Fb
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/bottom/dTdz"] = dTdz
    file["boundary_conditions/bottom/dbdz"] = dTdz * g * βT
    return nothing
end

filename = @sprintf("simple_flux_Fb%.0e_Fu%.0e_Nsq%.0e_Lz%d_Nz%d",
                    Fb, Fu, N², model.grid.Lz, model.grid.Nz)

u(model) = Array{Float32}(model.velocities.u.data.parent)
v(model) = Array{Float32}(model.velocities.v.data.parent)
w(model) = Array{Float32}(model.velocities.w.data.parent)
θ(model) = Array{Float32}(model.tracers.T.data.parent)

function calc_uw!(uw, u, w, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds uw[i, j, k] = ▶x_caa(i, j, k, grid, u) * ▶z_aac(i, j, k, grid, w)
            end
        end
    end

    return nothing
end

function calc_vw!(vw, v, w, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds vw[i, j, k] = ▶y_aca(i, j, k, grid, v) * ▶z_aac(i, j, k, grid, w)
            end
        end
    end

    return nothing
end

function calc_θw!(θw, θ, w, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds θw[i, j, k] = θ[i, j, k] * ▶z_aac(i, j, k, grid, w)
            end
        end
    end

    return nothing
end

function output_uw(model)
    uw = model.poisson_solver.storage
    u = model.velocities.u.data
    w = model.velocities.w.data
    @launch device(arch) config=launch_config(grid, 3) calc_uw!(uw, u, w, model.grid)
    return Array{Float32}(uw)
end

function output_vw(model)
    vw = model.poisson_solver.storage
    v = model.velocities.v.data
    w = model.velocities.w.data
    @launch device(arch) config=launch_config(grid, 3) calc_vw!(vw, v, w, model.grid)
    return Array{Float32}(vw)
end

function output_θw(model)
    θw = model.poisson_solver.storage
    θ = model.tracers.θ.data
    w = model.velocities.w.data
    @launch deθice(arch) config=launch_config(grid, 3) calc_θw!(θw, θ, w, model.grid)
    return Array{Float32}(θw)
end

fields = Dict(:u=>u, :θ=>θ, :w=>w, :θ=>θ) 
              #:uw=>output_uw, :vw=>output_vw, :θw=>output_θw)

field_writer = JLD2OutputWriter(model, fields; dir="data", prefix=filename,
                                init=init_savebcs, interval=1hour, force=true)

push!(model.output_writers, field_writer)

# 
# Run the simulation
#

if makeplot
    using PyPlot
    close("all")
    fig, axs = subplots(ncols=2, figsize=(10, 4))
end

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
    
    if makeplot
        sca(axs[1]); cla()
        imshow(rotr90(view(data(model.velocities.w), :, 2, :)))

        sca(axs[2]); cla()
        imshow(rotr90(view(data(model.velocities.u), :, 2, :)))
        show()
    end
end
