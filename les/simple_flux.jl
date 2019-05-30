using Oceananigans, Printf, PyPlot

include("utils.jl")

removespine(side; ax=gca()) = ax.spines[side].set_visible(false)
removespines(sides...; ax=gca()) = [removespine(side, ax=ax) for side in sides]

function makeplot(axs, model)

    wb = model.velocities.w * model.tracers.T
    wc = model.velocities.w * model.tracers.S
     e = turbulent_kinetic_energy(model)
     b = fluctuation(model.tracers.T)

    # Top row
    sca(axs[1, 1])
    cla()
    plot_xzslice(e, cmap="YlGnBu_r")
    title(L"e")

    sca(axs[1, 2])
    cla()
    plot_hmean(e, normalize=true)
    removespines("left", "top")
    axs[1, 2].tick_params(left=false, labelleft=false, right=true, labelright=true)
    xlim(-1, 1)

    # Middle row
    sca(axs[2, 1])
    cla()
    plot_xzslice(b, cmap="RdBu_r")
    title(L"b")

    sca(axs[2, 2])
    cla()
    plot_hmean(model.velocities.u, normalize=true)
    plot_hmean(model.velocities.v, normalize=true)
    removespines("left", "top")
    axs[2, 2].tick_params(left=false, labelleft=false, right=true, labelright=true)
    xlim(-1, 1)

    # Bottom row
    sca(axs[3, 1])
    cla()
    plot_xzslice(wc, cmap="RdBu_r")
    title(L"wc")

    sca(axs[3, 2])
    cla()
    plot_hmean(model.tracers.T, normalize=true)
    plot_hmean(model.tracers.S, normalize=true)
    plot_hmean(wb, normalize=true)
    removespines("left", "top")
    xlim(-1, 1)
    axs[3, 2].tick_params(left=false, labelleft=false, right=true, labelright=true)

    for ax in axs[1:3, 1]
        ax.axis("off")
        ax.set_aspect(1)
        ax.tick_params(left=false, labelleft=false, bottom=false, labelbottom=false)
    end

    tight_layout()

    return nothing
end

# 
# Model setup
# 

arch = CPU()
#@hascuda arch = GPU() # use GPU if it's available

model = Model(
     arch = arch, 
        N = (256,   1, 128), 
        L = (200, 200, 100), 
  closure = ConstantIsotropicDiffusivity(ν=1e-4, κ=1e-4),
      eos = LinearEquationOfState(βS=0.),
constants = PlanetaryConstants(f=0.)
)

#
# Initial condition, boundary condition, and tracer forcing
#

N² = 1e-8
Fb = 1e-8
Fu = 0.0

# Temperature initial condition
const dTdz = N² / (model.constants.g * model.eos.βT)
const T₀₀, h₀, δₕ = 20, -10, 2 # deg C

T₀★(z) = T₀₀ + dTdz * (z+h₀+δₕ) * step(z+h₀, δₕ) 

# Add a bit of surface-concentrated noise to the initial condition
ξ(z) = 1e-1 * rand() * exp(10z/model.grid.Lz) 
T₀(x, y, z) = T₀★(z) + dTdz*model.grid.Lz * ξ(z)

# Sponges
const μ₀ = 10 * Fb / N² / model.grid.Lz^2
const δˢ = model.grid.Lz / 10
const zˢ = -8 * model.grid.Lz / 10

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

c₀(x, y, z) = exp(z/δ) 
ν_∂z²_c★(z) = ν/δ^2 * c₀(0, 0, z)

#
# Set passive tracer forcing, boundary conditions and initial conditions
#

Base.@propagate_inbounds Fc(grid, u, v, w, T, S, i, j, k) = -ν_∂z²_c★(grid.zC[k])
model.forcing = Forcing(Fu=Fuˢ, Fv=Fvˢ, Fw=Fwˢ, FT=FTˢ, FS=Fc)

model.boundary_conditions.T.z.top = BoundaryCondition(Flux, Fb)
model.boundary_conditions.T.z.bottom = BoundaryCondition(Gradient, dTdz)

model.boundary_conditions.S.z.top = BoundaryCondition(Value, 1)
model.boundary_conditions.S.z.bottom = BoundaryCondition(Value, 0)

model.boundary_conditions.u.z.top = BoundaryCondition(Flux, Fu)

set_ic!(model, T=T₀, S=c₀)

fig, axs = subplots(ncols=2, nrows=3, sharey=true, figsize=(16, 6))

ρ₀ = 1035.0
cp = 3993.0

@printf("""
    Crunching a (viscous) ocean surface boundary layer with

        N : %d, %d, %d
       Fb : %.1e
       Fu : %.1e
        Q : %.2e
        τ : %.2e
        ν : %.1e

    Let's spin the gears.

    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, Fb, Fu, 
        -ρ₀*cp*Fb/(model.constants.g*model.eos.βT), ρ₀*Fu, model.closure.ν)

# Sensible initial time-step
αν = 1e-2
αu = 1e-1

for i = 1:100
    Δt = safe_Δt(model, αu, αν)
    walltime = @elapsed time_step!(model, 1, Δt)

    makeplot(axs, model)

    @printf("i: %d, t: %.2f hours, cfl: %.2e, wall: %s\n", model.clock.iteration,
            model.clock.time/3600, cfl(Δt, model), prettytime(1e9*walltime))
end
