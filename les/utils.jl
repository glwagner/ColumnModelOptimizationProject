@hascuda using CuArrays, CUDAnative, CUDAdrv
using GPUifyLoops: @launch, @loop
using Oceananigans: device, launch_config, Diagnostic, cell_advection_timescale
using FileIO: save

import Oceananigans: run_diagnostic, time_to_run

struct FieldOutput{O, F}
    outputtype :: O
    field :: F
end

FieldOutput(field) = FieldOutput(Array, field) # default
(fo::FieldOutput)(model) = fo.outputtype(fo.field.data.parent)

function FieldOutputs(fields)
    names = propertynames(fields)
    nfields = length(fields)
    return Dict((names[i], FieldOutput(fields[i])) for i in 1:nfields)
end

#
# Cell diffusion timescale
#

function cell_diffusion_timescale(model::Model{TS, <:AnisotropicMinimumDissipation}) where TS
    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    max_ν = maximum(model.diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in model.diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

function cell_diffusion_timescale(model)
    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    max_ν = maximum(model.diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
end

#
# Accumulated diagnostics
#

abstract type AccumulatedDiagnostic <: Diagnostic end

run_diagnostic(model, a::AccumulatedDiagnostic) = push!(a.data, a(model))
time_to_run(clock::Clock, a::AccumulatedDiagnostic) = (clock.iteration % a.frequency) == 0

function save_accumulated_diagnostics!(filepath, names, model)
    rm(filepath, force=true)
    save(filepath, Dict(names[i]=>d.data for (i, d) in enumerate(model.diagnostics)))
    return nothing
end

# 
# Time
#

struct TimeDiagnostic{T} <: AccumulatedDiagnostic
    frequency :: Int
    data :: Vector{T}
end

TimeDiagnostic(T=Float64; frequency=1) = TimeDiagnostic(frequency, T[])
(::TimeDiagnostic)(model) = model.clock.time

#
# CFL diagnostic
#

struct CFL{DT, T, TS} <: AccumulatedDiagnostic
    frequency :: Int
    Δt :: DT
    data :: Vector{T}
    timescale :: TS
end

(c::CFL{<:Number})(model) = c.Δt / c.timescale(model)
(c::CFL{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

AdvectiveCFL(Δt; frequency=1) = 
    CFL(frequency, Δt, eltype(model.grid)[], cell_advection_timescale)

DiffusiveCFL(Δt; frequency=1) = 
    CFL(frequency, Δt, eltype(model.grid)[], cell_diffusion_timescale)

timescalename(::typeof(cell_advection_timescale)) = "Advective"
timescalename(::typeof(cell_diffusion_timescale)) = "Diffusive"
diagname(c::CFL) = timescalename(c.timescale) * "CFL"

#
# Max diffusivity diagnostic
#

struct MaxAbsFieldDiagnostic{T, F} <: AccumulatedDiagnostic
    frequency :: Int
    data :: Vector{T}
    field :: F
end

function MaxAbsFieldDiagnostic(field; frequency=1) 
    T = typeof(maximum(abs, field.data.parent))
    MaxAbsFieldDiagnostic(frequency, T[], field) 
end

(m::MaxAbsFieldDiagnostic)(model) = maximum(abs, m.field.data.parent)

#
# Max vertical variance diagnostic
#

#=
struct MaxWsqDiagnostic{T} <: AccumulatedDiagnostic
    frequency :: Int
    data :: Vector{T}
end

MaxWsqDiagnostic(T=Float64; frequency=1) = MaxWsqDiagnostic(frequency, T[])
(c::MaxWsqDiagnostic)(model) = max_vertical_velocity_variance(model)
diagname(::MaxWsqDiagnostic) = "MaxWsq"

function max_vertical_velocity_variance(model)
    @launch device(model.arch) config=launch_config(model.grid, 3) w²!(model.poisson_solver.storage,
                                                                       model.velocities.w.data, model.grid)
    return maximum(real, model.poisson_solver.storage)
end

function w²!(w², w, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds w²[i, j, k] = w[i, j, k]^2
            end
        end
    end
    return nothing
end

Base.typemin(::Type{Complex{T}}) where T = T
=#
