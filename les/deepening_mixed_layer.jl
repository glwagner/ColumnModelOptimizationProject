using Oceananigans, Random, Printf, JLD2, Statistics, UnicodePlots

using Oceananigans.TurbulenceClosures: AbstractSmagorinsky

@hascuda using CuArrays

include("utils.jl")

# 
# Model set-up
#

# Two cases from Van Roekel et al (JAMES, 2018)
parameters = Dict(
    :free_convection => Dict(:Qb=>3.39e-8, :Qu=>0.0,      :f=>1e-4, :N²=>1.96e-5, :tf=>8day),
    :wind_stress     => Dict(:Qb=>0.0,     :Qu=>-9.66e-5, :f=>0.0,  :N²=>9.81e-5, :tf=>8day)
)

# Simulation parameters
case = :wind_stress
Nx = 128
Nz = 256            # Resolution    
Lx = Lz = 128       # Domain extent
Δt = 0.5

N², Qb, Qu, f, tf = (parameters[case][p] for p in (:N², :Qb, :Qu, :f, :tf))
αθ, g = 2e-4, 9.81
Qθ, dθdz = Qb / (g*αθ), N² / (g*αθ)

# Create boundary conditions.
ubcs = HorizontallyPeriodicBCs(top=BoundaryCondition(Flux, Qu))
θbcs = HorizontallyPeriodicBCs(top=BoundaryCondition(Flux, Qθ), 
                               bottom=BoundaryCondition(Gradient, dθdz))

# Instantiate the model
model = Model(      arch = HAVE_CUDA ? GPU() : CPU(), 
                       N = (Nx, Nx, Nz), 
                       L = (Lx, Lx, Lz),
                     eos = LinearEquationOfState(βT=αθ, βS=0.0),
               constants = PlanetaryConstants(f=f, g=g),
                 #closure = VerstappenAnisotropicMinimumDissipation(C=1/12),
                 #closure = ConstantSmagorinsky(),
                 closure = BlasiusSmagorinsky(),
                     bcs = BoundaryConditions(u=ubcs, T=θbcs))

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
θᵢ(x, y, z) = 20 + dθdz * z + 1e-3 * dθdz * model.grid.Lz * Ξ(z)
uᵢ(x, y, z) = 1e-3 * Ξ(z)
set!(model, u=uᵢ, v=uᵢ, w=uᵢ, T=θᵢ)

T_dev = Array{Float64}(undef, 1, 1, model.grid.Tz)
@hascuda T_dev = CuArray{Float64}(undef, 1, 1, model.grid.Tz)

function plot_average_temperature(model)
    T_dev .= mean(model.tracers.T.data.parent, dims=(1, 2))
    T = Array(T_dev)
    return lineplot(T[2:end-1], model.grid.zC, height=40, canvas=DotCanvas, 
                    xlim=[20-dθdz*Lz, 20], ylim=[-Lz, 0])
end

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.4, Δt=Δt, max_change=1.1, max_Δt=Δt)

#
# Set up output
#

function init_bcs(file, model)
    file["boundary_conditions/top/Qb"] = Qb
    file["boundary_conditions/top/Qθ"] = Qθ
    file["boundary_conditions/top/Qu"] = Qu
    file["boundary_conditions/bottom/N²"] = N²
    return nothing
end

u(model) = Array(model.velocities.u.data.parent)
v(model) = Array(model.velocities.v.data.parent)
w(model) = Array(model.velocities.w.data.parent)
T(model) = Array(model.tracers.T.data.parent)
νₑ(model) = Array(model.diffusivities.νₑ.data.parent)
κₑ(model::Model{TS, <:VerstappenAnisotropicMinimumDissipation}) where TS = 
	Array(model.diffusivities.κₑ.T.data.parent)
κₑ(model::Model{TS, <:AbstractSmagorinsky}) where TS = 0.0

closurename(closure::VerstappenAnisotropicMinimumDissipation) = @sprintf("amd%.2f", closure.C)
closurename(closure::BlasiusSmagorinsky) = @sprintf("bsmag%.2f", closure.C)
closurename(closure::ConstantSmagorinsky) = @sprintf("dsmag%.2f", closure.Cs)

fields = Dict(:u=>u, :v=>v, :w=>w, :T=>T, :ν=>νₑ, :κ=>κₑ)
filename = @sprintf("%s_Nx%d_Nz%d_%s_dt%.2f", case, Nx, Nz, closurename(model.closure), Δt)
field_writer = JLD2OutputWriter(model, fields; dir="data", init=init_bcs, prefix=filename, 
                                max_filesize=1GiB, interval=6hour, force=true)
push!(model.output_writers, field_writer)

#
# Set up diagnostics
#

frequency = 10

diff_cfl = DiffusiveCFL(wizard, frequency=frequency)
adv_cfl = AdvectiveCFL(wizard, frequency=frequency)
max_u = MaxAbsFieldDiagnostic(model.velocities.u, frequency=frequency)
max_v = MaxAbsFieldDiagnostic(model.velocities.v, frequency=frequency)
max_w = MaxAbsFieldDiagnostic(model.velocities.w, frequency=frequency)
max_ν = MaxAbsFieldDiagnostic(model.diffusivities.νₑ, frequency=frequency)
max_κ = MaxAbsFieldDiagnostic(model.diffusivities.κₑ.T, frequency=frequency)
w² = MaxWsqDiagnostic(frequency=frequency)
tdiag = TimeDiagnostic(frequency=frequency)

if typeof(model.clousre) <: VerstappenAnisotropicMinimumDissipation
	push!(model.diagnostics, max_w, adv_cfl, diff_cfl, max_u, max_v, w², max_ν, max_κ, tdiag)
	diag_names = ("max_w", "adv_cfl", "diff_cfl", "max_u", "max_v", "wsq", "max_nu", "max_kappa", "t")
else
	push!(model.diagnostics, max_w, adv_cfl, diff_cfl, max_u, max_v, w², max_ν, tdiag)
	diag_names = ("max_w", "adv_cfl", "diff_cfl", "max_u", "max_v", "wsq", "max_nu", "t")
end

diag_filepath = joinpath("data", filename * "_diags.jld2")

# 
# Run the simulation
#

terse_message(model, walltime, Δt) =
    @sprintf(
    "i: %d, t: %.4f hours, Δt: %.3f s, wmax: %.6f ms⁻¹, adv cfl: %.3f, diff cfl: %.3f, wall time: %s\n",
    model.clock.iteration, model.clock.time/3600, Δt, 
    model.diagnostics[1].data[end], 
    model.diagnostics[2].data[end],
    model.diagnostics[3].data[end],
    prettytime(walltime)
   )

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = Base.@elapsed time_step!(model, 100, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    if model.clock.iteration % 10000 == 0
        save_accumulated_diagnostics!(diag_filepath, diag_names, model)

        plt = plot_average_temperature(model)
        show(plt)
        @printf "\n"
    end
end
