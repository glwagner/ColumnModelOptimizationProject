using Pkg; Pkg.activate("..")
using OceanTurb, ColumnModelOptimizationProject.KPPOptimization, Printf

stdkwargs = Dict(:N=>600, :L=>150, :h₀=>20, :d=>4)

models = Dict(
    "unstable_weak"   => simple_flux_model(dBdz = 2.5e-6, Fb = 5e-9,   Fu = -5e-4, stdkwargs...),
    "unstable_strong" => simple_flux_model(dBdz = 2.5e-5, Fb = 5e-9,   Fu = -5e-4, stdkwargs...),
    "stable_weak"     => simple_flux_model(dBdz = 2.5e-6, Fb = -1e-10, Fu = -1e-4, stdkwargs...),
    "stable_strong"   => simple_flux_model(dBdz = 2.5e-7, Fb = -5e-10, Fu = -5e-5, stdkwargs...),
    "neutral"         => simple_flux_model(dBdz = 2.5e-6, Fb = 0,      Fu = -5e-4, stdkwargs...)
)

for (case, model) in models
    @printf "Generating data for case %s..." case
    filepath = joinpath("..", "data", "perfect_model", "$case.jld2")

    isfile(filepath) && rm(filepath)

    @time generate_data(filepath, model; dout=1hour, tfinal=2day)
end
