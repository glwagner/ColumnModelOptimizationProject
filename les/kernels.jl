using Oceananigans.TurbulenceClosures: ▶xz_caf, ▶yz_acf, ▶z_aaf, ▶x_faa, ▶y_afa, 
                                       ν_Σᵢⱼ_cff, ν_Σᵢⱼ_fcf, ν_Σᵢⱼ_ccc, κ_∂z_c, Σ₁₃, Σ₂₃, Σ₃₃

using Oceananigans: datatuple, datatuples, PressureBoundaryConditions

#####
##### Advective fluxes
#####

function calculate_wϕ!(wϕ, grid, w, ϕ, ▶w, ▶ϕ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds wϕ[i, j, k] = ▶w(i, j, k, grid, w) * ▶ϕ(i, j, k, grid, ϕ)
            end
        end
    end
    return nothing
end

function prepare_advective_flux_calculation(model)
    arch, grid = model.arch, model.grid
    U, Φ = datatuples(model.velocities, model.tracers)
    #bcs_args = (model.clock.time, model.clock.iteration, U, Φ)
    #fill_halo_regions!(merge(U, Φ), model.boundary_conditions, arch, grid, bcs_args...)
    return U, Φ, arch, grid
end

function calculate_wu!(wu, model)
    U, Φ, arch, grid = prepare_advective_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wu, grid, U.w, U.u, ▶z_aaf, ▶x_faa)
    return nothing
end

function calculate_wv!(wv, model)
    U, Φ, arch, grid = prepare_advective_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wv, grid, U.w, U.v, ▶z_aaf, ▶y_afa)
    return nothing
end

function calculate_wθ!(wθ, model)
    U, Φ, arch, grid = prepare_advective_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_wϕ!(wθ, grid, U.w, Φ.T, ▶z_aaf, nointerp)
    return nothing
end

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


function prepare_diffusive_flux_calculation(model)
    arch, grid = model.arch, model.grid
    U, Φ, K = datatuples(model.velocities, model.tracers, model.diffusivities)
    #bcs_args = (model.clock.time, model.clock.iteration, U, Φ)
    #fill_halo_regions!(merge(U, Φ), model.boundary_conditions, arch, grid, bcs_args...)
    return U, Φ, K, arch, grid
end

function calculate_νΣ₁₃!(νΣ, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, K.νₑ, Σ₁₃, ν_Σᵢⱼ_fcf)
    return nothing
end

function calculate_νΣ₂₃!(νΣ, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, K.νₑ, Σ₂₃, ν_Σᵢⱼ_cff)
    return nothing
end

function calculate_νΣ₃₃!(νΣ, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_νΣᵢⱼ!(νΣ, grid, U.u, U.v, U.w, K.νₑ, Σ₃₃, ν_Σᵢⱼ_ccc)
    return nothing
end

function calculate_κθz!(κθz, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_κϕz!(κθz, grid, K.κₑ.T, Φ.T)
    return nothing
end

#####
##### Total fluxes
#####

function calculate_momentum_flux!(flux, grid, u, v, w, ν, ϕ, ▶w, ▶ϕ, Σ, ν_Σᵢⱼ_aaa)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds flux[i, j, k] = ▶w(i, j, k, grid, w) * ▶ϕ(i, j, k, grid, ϕ) - 2 * ν_Σᵢⱼ_aaa(i, j, k, grid, ν, Σ, u, v, w)
            end
        end
    end
    return nothing
end

function calculate_tracer_flux!(flux, grid, w, κ, ϕ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds flux[i, j, k] = w[i, j, k] * ▶z_aaf(i, j, k, grid, ϕ) - κ_∂z_c(i, j, k, grid, ϕ, κ, nothing)
            end
        end
    end
    return nothing
end

function calculate_qu!(flux, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_momentum_flux!(flux, grid, U.u, U.v, U.w, K.νₑ, U.u, 
                                                                                ▶x_faa, ▶z_aaf, Σ₁₃, ν_Σᵢⱼ_fcf)
    return nothing
end

function calculate_qv!(flux, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_momentum_flux!(flux, grid, U.u, U.v, U.w, K.νₑ, U.v, 
                                                                                ▶y_afa, ▶z_aaf, Σ₂₃, ν_Σᵢⱼ_cff)
    return nothing
end

function calculate_qθ!(flux, model)
    U, Φ, K, arch, grid = prepare_diffusive_flux_calculation(model)
    @launch device(arch) config=launch_config(grid, 3) calculate_tracer_flux!(flux, grid, U.w, K.κₑ.T, Φ.T) 
    return nothing
end

