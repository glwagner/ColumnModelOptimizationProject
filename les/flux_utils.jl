
cfl(Δt, model) = Δt * Umax(model) / Δmin(model.grid)

get_ν(c::ConstantSmagorinsky) = c.ν_background
get_ν(c::AnisotropicMinimumDissipation) = c.ν_background
get_ν(c) = c.ν

function safe_Δt(model, αu, αν=0.01)
    τu = Δmin(model.grid) / Umax(model)
    τν = Δmin(model.grid)^2 / get_ν(model.closure)

    return min(αν*τν, αu*τu)
end
