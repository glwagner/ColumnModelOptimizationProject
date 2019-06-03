using Distributed

using JLD2

mutable struct JLD2OutputWriter{O} <: OutputWriter
            filepath :: String
             outputs :: O
    output_frequency :: Int
        asynchronous :: Bool
end

function savesubstruct!(file, model, name, flds=propertynames(getproperty(model, name)))
    for fld in flds
        file["$name/$fld"] = getproperty(getproperty(model, name), fld)
    end
    return nothing
end

noinit(args...) = nothing

function JLD2OutputWriter(model, outputs; dir=".", prefix="", frequency=1, init=noinit, force=false,
                          asynchronous=false)

    mkpath(dir)
    filepath = joinpath(dir, prefix*".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    jldopen(filepath, "a+") do file
        init(file, model)
        savesubstruct!(file, model, :grid)
        savesubstruct!(file, model, :eos)
        savesubstruct!(file, model, :constants)
        savesubstruct!(file, model, :closure)
    end

    return JLD2OutputWriter(filepath, outputs, frequency, asynchronous)
end

function Oceananigans.write_output(model, fw::JLD2OutputWriter)
    @info @sprintf("Calculating JLD2 output %s...", keys(fw.outputs))
    @time data = Dict((name, f(model)) for (name, f) in fw.outputs)

    iter = model.clock.iteration
    time = model.clock.time
    path = fw.filepath

    @info @sprintf("Writing JLD2 output %s...", keys(fw.outputs))
    t0 = time_ns()
    if fw.asynchronous
        @async remotecall(jld2output!, 2, path, iter, time, data)
    else
        jld2output!(path, iter, time, data)
    end
    @info "Done writing (t: $(prettytime(time_ns()-t0))"

    return nothing
end

function jld2output!(filepath, iter, time, data)
    jldopen(filepath, "r+") do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return nothing
end
