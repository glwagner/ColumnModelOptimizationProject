get_cfl(Δt, model) = Δt * Umax(model) / Δmin(model.grid)

function safe_Δt(model, αu, αν=0.01)
    τu = Δmin(model.grid) / Umax(model)
    τν = Δmin(model.grid)^2 / model.closure.ν

    return min(αν*τν, αu*τu)
end

function cfl_Δt(model, cfl, max_Δt)
    τ = Δmin(model.grid) / Umax(model)
    return min(max_Δt, cfl*τ)
end

Base.@kwdef mutable struct CFLUtility{T}
           cfl :: T = 0.1
            rν :: T = 2e-2
    max_change :: T = 2.0
    min_change :: T = 0.5
        max_Δt :: T = Inf
            Δt :: T = 1.0
end

function new_Δt(model, cfl_util)
    new_Δt_u = cfl_util.cfl * Δmin(model.grid) / Umax(model)
    new_Δt_ν = cfl_util.rν * Δmin(model.grid)^2 / model.closure.ν

    if isnan(new_Δt)
        new_Δt = cfl_util.Δt
    end

    new_Δt = min(cfl_util.max_change * cfl_util.Δt, new_Δt)
    new_Δt = max(cfl_util.min_change * cfl_util.Δt, new_Δt)
    new_Δt = min(max_Δt, new_Δt)

    cfl_util.Δt = Δt

    return Δt
end
