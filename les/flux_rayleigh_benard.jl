using Oceananigans, Printf, JLD2

# 
# Model set-up
#

# Simulation parameters
N = 256
Qb = 1.0
Pr = 0.7
Ra = 10^4 # Ra = Qb * Lz^4 * Pr^2 / ν^3
ν = (Pr^2 * Qb / Ra)^(1/3)
κ = ν / Pr
tf = 10000

# Create boundary conditions.
bbcs = HorizontallyPeriodicBCs(top=BoundaryCondition(Flux, Qb/2), 
                               bottom=BoundaryCondition(Flux, Qb/2))

# Instantiate the model
model = Model(      arch = HAVE_CUDA ? GPU() : CPU(), 
                       N = (2N, 1, N), 
                       L = (2, 2, 1),
                     eos = LinearEquationOfState(βT=1.0, βS=0.0),
               constants = PlanetaryConstants(f=0.0, g=1.0),
                 closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
                 #closure = VerstappenAnisotropicMinimumDissipation(ν=ν, κ=κ, C=1/12),
                     bcs = BoundaryConditions(T=bbcs))

Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
bᵢ(x, y, z) = -Qb / κ * z + 1e1 * Ξ(z)
uᵢ(x, y, z) = 1e1 * Ξ(z)
wᵢ(x, y, z) = 1e1 * Ξ(z)
set!(model, u=uᵢ, w=uᵢ, T=bᵢ)

#
# Set up output
#

init(file, model) = file["boundary_conditions/top/Qb"] = Qb
u(model) = Array(model.velocities.u.data.parent)
v(model) = Array(model.velocities.v.data.parent)
w(model) = Array(model.velocities.w.data.parent)
T(model) = Array(model.tracers.T.data.parent)

const VAMD = VerstappenAnisotropicMinimumDissipation
const CID = ConstantIsotropicDiffusivity

νₑ(model::Model{TS, <:CID}) where TS = model.closure.ν
νₑ(model) = Array(model.diffusivities.νₑ.data.parent)   
κₑ(model) = model.closure.κ
κₑ(model::Model{TS, <:VAMD}) where TS = Array(model.diffusivities.κₑ.T.data.parent)   

closurename(::CID) = "DNS"
closurename(closure::VAMD) = @sprintf("AMD_C%.2f", closure.C)

fields = Dict(:u=>u, :v=>v, :w=>w, :T=>T, :ν=>νₑ, :κ=>κₑ)
filename = @sprintf("flux_rb_Ra%d_N%d_%s", Ra, N, closurename(model.closure))
field_writer = JLD2OutputWriter(model, fields; dir="data", init=init, prefix=filename, 
                                max_filesize=200MiB, interval=1000, force=true)
push!(model.output_writers, field_writer)

# 
# Run the simulation
#

function terse_message(model, walltime, Δt)
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)
    return @sprintf("i: %d, t: %.4f, Δt: %.3f, wmax: %.6f, cfl: %.3f, wall time: %s\n",
                    model.clock.iteration, model.clock.time/3600, Δt, wmax, cfl, prettytime(walltime))
end

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.1, Δt=1e-5, max_change=1.1, max_Δt=1e-3)

w²_filename = @sprintf("data/%s_wsq.jld2", filename)
max_w², t = [], []
include("velocity_variance_utils.jl")

# Spin up
while model.clock.iteration < 1000
    update_Δt!(wizard, model)
    walltime = Base.@elapsed step_with_w²!(max_w², t, model, wizard.Δt, 2)
    model.clock.iteration % 10 == 0 && @printf "%s" terse_message(model, walltime, wizard.Δt)
end

wizard.cfl = 0.05
using PyPlot
fig, axs = subplots()

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = Base.@elapsed step_with_w²!(max_w², t, model, wizard.Δt, 100)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    rm(w²_filename, force=true)
    @save w²_filename max_w² t

    sca(axs); cla()
    imshow(rotr90(view(model.tracers.T.data.parent, :, 3, :)))

end
