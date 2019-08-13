using GPUifyLoops: @launch, @loop, @unroll
using Oceananigans: launch_config

function w²!(w², w, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                w²[i, j, k] = w[i, j, k]^2
            end
        end
    end
    return nothing
end

function compute_and_store_w²!(max_w², model)
    @launch device(model.arch) config=launch_config(model.grid, 3) w²!(model.poisson_solver.storage, 
                                                                       model.velocities.w, model.grid)
    push!(max_w², maximum(abs, model.poisson_solver.storage))
    return nothing
end

function step_with_w²!(max_w², model, Δt, Nt)
    time_step!(model, 1, Δt)
    compute_and_store_w²!(max_w², model)

    for i = 2:Nt
        time_step!(model, 1, Δt, init_with_euler=false)
        compute_and_store_w²!(max_w², model)
    end

    return nothing
end

step_with_w²!(max_w²::Nothing, model, Δt, Nt) = time_step!(model, Nt, Δt)
