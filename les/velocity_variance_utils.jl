using CuArrays, CUDAnative, CUDAdrv
using GPUifyLoops: @launch, @loop
using Oceananigans: device, launch_config

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

function compute_and_store_w²!(max_w², t, model)
    @launch device(model.arch) config=launch_config(model.grid, 3) w²!(model.pressures.pHY′.data,
                                                                       model.velocities.w.data, model.grid)
    push!(max_w², maximum(model.pressures.pHY′.data.parent))
    push!(t, model.clock.time)
    return nothing
end

function step_with_w²!(max_w², t, model, Δt, Nt)
    time_step!(model, 1, Δt)
    compute_and_store_w²!(max_w², model)

    for i = 2:Nt
        time_step!(model, 1, Δt, init_with_euler=false)
        compute_and_store_w²!(max_w², t, model)
    end

    return nothing
end

step_with_w²!(max_w²::Nothing, t, model, Δt, Nt) = time_step!(model, Nt, Δt)
