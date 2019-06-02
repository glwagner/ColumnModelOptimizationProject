using Distributed
addprocs(1)

using Oceananigans, Printf, PyPlot

include("utils.jl")

#
# Initial condition, boundary condition, and tracer forcing
#

 N = 16
Pr = 0.7
Re = 10^3
Ri = -0.1
 L = 1.0
ΔU = 1.0

# Computed parameters
Δb = Ri * ΔU^2 / L
 ν = ΔU * L / Re
 κ = ν / Pr

Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value,  Δb),
    bottom = BoundaryCondition(Value, -Δb)
   ))

ubcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value,  ΔU),
    bottom = BoundaryCondition(Value, -ΔU),
   ))

# 
# Model setup
# 

arch = CPU()
#@hascuda arch = GPU() # use GPU if it's available

model = Model(
         arch = arch, 
            N = (8N, 1, 8N),
            L = (2L, L,  L),
      closure = AnisotropicMinimumDissipation(C=0.3, ν_background=ν, κ_background=κ),
          eos = LinearEquationOfState(βT=1.0, βS=0.),
    constants = PlanetaryConstants(f=0.0, g=1.0),
          bcs = BoundaryConditions(u=ubcs, T=Tbcs)
    )

filename(model) = @sprintf("stratified_couette_Re%d_Ri%.3f_Nz%d", Re, Ri, model.grid.Nz)

# Add a bit of surface-concentrated noise to the initial condition
ξ(z) = 1e-1 * rand() * z/model.grid.Lz * (z/model.grid.Lz + 1)

T₀(x, y, z) = 2Δb * (1/2 + z/model.grid.Lz) * (1 + ξ(z))
u₀(x, y, z) = 2ΔU * (1/2 + z/model.grid.Lz) * (1 + ξ(z))
v₀(x, y, z) = 1e-6 * ξ(z)
w₀(x, y, z) = ξ(z)
c₀(x, y, z) = ξ(z)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=c₀)

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
b(model)  = Array(data(model.tracers.T))

fields = Dict(:u=>u, :v=>v, :w=>w, :b=>b, :c=>c)

field_writer = JLD2OutputWriter(model, fields; dir="data", 
                                prefix=filename(model)*"_fields", 
                                init=savebcs, frequency=1000, force=true,
                                asynchronous=true)

push!(model.output_writers, field_writer)
=#

@printf(
    """
    Crunching stratified Couette flow with
    
            n : %d, %d, %d
           Re : %.0e
           Ri : %.1e
    
    Let's spin the gears.
    
    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, Re, Ri
             
)

gridspec = Dict("width_ratios"=>[Int(model.grid.Lx/model.grid.Lz)+1, 1])
fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(8, 10), gridspec_kw=gridspec)

# Sensible initial time-step
αν = 1e-1
αu = 1e-1

# Spinup
for i = 1:100
    Δt = safe_Δt(model, αu, αν)
    walltime = @elapsed time_step!(model, 1, Δt)
end

# Main loop
for i = 1:100
    Δt = safe_Δt(model, αu, αν)
    walltime = @elapsed time_step!(model, 1000, Δt)

    channelplot(axs, model)

    @printf("i: %d, t: %.2e, Δt: %.2e s, cfl: %.2e, wall: %s\n", 
            model.clock.iteration, model.clock.time/3600, Δt,
            cfl(Δt, model), prettytime(1e9*walltime))
end
