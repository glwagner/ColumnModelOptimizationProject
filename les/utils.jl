@hascuda using CuArrays, CUDAnative, CUDAdrv
using GPUifyLoops: @launch, @loop
using Oceananigans: device, launch_config, Diagnostic, cell_advection_timescale
using FileIO: save

import Oceananigans: run_diagnostic, time_to_run

include("kernels.jl")

mutable struct HorizontallyAveragedFlux{K, H, P, I, Ω, R} <: Diagnostic
       calculate_flux! :: K
    horizontal_average :: H
               profile :: P
              interval :: I
             frequency :: Ω
              previous :: Float64
           return_type :: R
end

function HorizontallyAveragedFlux(model, flux_name; interval=nothing, frequency=nothing, return_type=Array)
    kernel_name = Symbol(:calculate_, flux_name, :!)
    calculate_flux! = eval(kernel_name)
    havg = HorizontalAverage(model, model.pressures.pHY′, frequency=1, return_type=return_type)
    return HorizontallyAveragedFlux(calculate_flux!, havg, havg.profile, interval, frequency, 0.0, return_type)
end

function run_diagnostic(model, flux_avg::HorizontallyAveragedFlux)
    flux_avg.calculate_flux!(model.pressures.pHY′.data, model) 
    run_diagnostic(model, flux_avg.horizontal_average)
    return nothing
end

const HAF = HorizontallyAveragedFlux
Base.getproperty(fluxavg::HAF, name::Symbol) = get_fluxavg_property(fluxavg, Val(name))
get_fluxavg_property(fluxavg::HAF, ::Val{N}) where N = getfield(fluxavg, N)
get_fluxavg_property(fluxavg::HAF, ::Val{:profile}) = fluxavg.horizontal_average.profile

#
# Time averaging...
#

mutable struct TimeAndHorizontalAverage{T, H, Ω, R} <: Diagnostic
            time_average :: T
      horizontal_average :: H
               frequency :: Ω
             return_type :: R
    averaging_start_time :: Float64
    increment_start_time :: Float64
                previous :: Float64
end

function TimeAndHorizontalAverage(model, havg; return_type=Array)
    profile = zeros(model.arch, model.grid, 1, 1, model.grid.Tz)
    time_average = TimeAndHorizontalAverage(profile, havg, 1, return_type, 0.0, 0.0, 0.0)
    push!(model.diagnostics, time_average)
    return time_average
end

function TimeAveragedField(model, fields::Union{Field, Tuple}; return_type=Array)
    havg = HorizontalAverage(model, fields, frequency=1, return_type=return_type)
    return TimeAndHorizontalAverage(model, havg; return_type=return_type)
end

function TimeAveragedFlux(model, flux_name; return_type=Array)
    havg = HorizontallyAveragedFlux(model, flux_name, frequency=1, return_type=return_type)
    return TimeAndHorizontalAverage(model, havg; return_type=return_type)
end

function run_diagnostic(model, tavg::TimeAndHorizontalAverage)
    if tavg.increment_start_time == tavg.averaging_start_time
        # First increment: zero out time-averaged profile
        tavg.time_average .= 0
    end

    # Compute current horizontal average
    run_diagnostic(model, tavg.horizontal_average)

    # Add increment to time-averaged profile
    Δt = model.clock.time - tavg.increment_start_time
    @. tavg.time_average += tavg.horizontal_average.profile * Δt

    # Reset increment start time
    tavg.increment_start_time = model.clock.time
    return nothing
end

returnit(::Nothing, obj) = obj
returnit(return_type, obj) = return_type(obj)

function (tavg::TimeAndHorizontalAverage)(model)
    # Compute total interval duration
    ΔT = model.clock.time - tavg.averaging_start_time

    # Compute average assuming that time_average contains the time integral.
    tavg.time_average ./= ΔT

    # Reset
    tavg.averaging_start_time = model.clock.time
    tavg.increment_start_time = model.clock.time

    return returnit(tavg.return_type, tavg.time_average)
end

#
# Time dependent boundary conditions
#

struct TimeDependentBoundaryCondition{C} <: Function
    c :: C
end

TimeDependentBoundaryCondition(Tbc, c) = BoundaryCondition(Tbc, TimeDependentBoundaryCondition(c))

@inline (bc::TimeDependentBoundaryCondition)(i, j, grid, time, args...) = bc.c(time)

struct FieldOutput{O, F}
    return_type :: O
          field :: F
end

FieldOutput(field) = FieldOutput(Array, field) # default
(fo::FieldOutput)(model) = fo.return_type(fo.field.data.parent)

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
