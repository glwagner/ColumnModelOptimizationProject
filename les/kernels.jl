using Oceananigans.TurbulenceClosures: ▶xz_caf, ▶yz_acf, ▶z_aaf, ▶x_faa, ▶y_afa, 
                                       ν_Σᵢⱼ_cff, ν_Σᵢⱼ_fcf, ν_Σᵢⱼ_ccc, κ_∂z_c, Σ₁₃, Σ₂₃, Σ₃₃

#####
##### Subgrid-scale fluxes
#####

@inline nointerp(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]

function calculate_νΣᵢⱼ!(νΣ, grid, u, v, w, ν, Σ, ν_Σᵢⱼ_aaa)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds νΣ[i, j, k] = ν_Σᵢⱼ_aaa(i, j, k, grid, ν, Σ, u, v, w)
            end
        end
    end
    return nothing
end

function calculate_κϕz!(κϕz, grid, κ, ϕ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds κϕz[i, j, k] = κ_∂z_c(i, j, k, grid, ϕ, κ, nothing)
            end
        end
    end
    return nothing
end


function calculate_νΣ₁₃!(νΣ, model)
    arch, grid = model.arch, model.grid
    U, K  = datatuples(model.velocities, model.diffusivities)  
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, U.νₑ, Σ₁₃, ν_Σᵢⱼ_fcf)
    return nothing
end

function calculate_νΣ₂₃!(νΣ, model)
    arch, grid = model.arch, model.grid
    U, K  = datatuples(model.velocities, model.diffusivities)  
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, U.νₑ, Σ₂₃, ν_Σᵢⱼ_cff)
    return nothing
end

function calculate_νΣ₃₃!(νΣ, model)
    arch, grid = model.arch, model.grid
    U, K  = datatuples(model.velocities, model.diffusivities)  
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, U.νₑ, Σ₃₃, ν_Σᵢⱼ_ccc)
    return nothing
end

function calculate_κθz!(κθz, model)
    arch, grid = model.arch, model.grid
    Φ, K  = datatuples(model.tracers, model.diffusivities)  
    @launch device(arch) config=launch_config(grid, 3) calculate_κϕz!(κθz, grid, K.κₑ.T, Φ.T)
    return nothing
end

#####
##### Advective fluxes
#####

function calculate_wϕ!(wϕ, grid, w, ϕ, ▶w, ▶u)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds wϕ[i, j, k] = ▶w(i, j, k, grid, w) * ▶ϕ(i, j, k, grid, ϕ)
            end
        end
    end
    return nothing
end

function calculate_wu!(wu, model)
    arch, grid = model.arch, model.grid
    U = datatuple(model.velocities)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wu, grid, U.w, U.u, ▶x_faa, ▶z_aaf) 
    return nothing
end

function calculate_wv!(wv, model)
    arch, grid = model.arch, model.grid
    U = datatuple(model.velocities)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wv, grid, U.w, U.v, ▶y_afa, ▶z_aaf) 
    return nothing
end

function calculate_wT!(wT, model)
    arch, grid = model.arch, model.grid
    U, Φ = datatuple(model.velocities, model.tracers)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wT, grid, U.w, Φ.T, nointerp, ▶z_aaf) 
    return nothing
end
