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
          νcfl :: T = 2e-2
    max_change :: T = 2.0
    min_change :: T = 0.5
        max_Δt :: T = Inf
            Δt :: T = 1.0
end

nan2inf(a) = isnan(a) ? Inf : a

function new_Δt(model, util)
    Δt_velocity = util.cfl * Δmin(model.grid) / Umax(model)
    Δt_diffusivity = util.νcfl * Δmin(model.grid)^2 / max(model.closure.ν, model.closure.κ)

    Δt = min(nan2inf(Δt_velocity), 
             nan2inf(Δt_diffusivity))

    Δt = min(util.max_change * util.Δt, Δt)
    Δt = max(util.min_change * util.Δt, Δt)
    Δt = min(util.max_Δt, Δt)

    util.Δt = Δt

    return Δt
end
