using JLD2

mutable struct JLD2OutputWriter{O} <: OutputWriter
            filepath :: String
             outputs :: O
    output_frequency :: Int
end

function savesubstruct!(file, model, name, flds=propertynames(getproperty(model, name)))
    for fld in flds
        file["$name/$fld"] = getproperty(getproperty(model, name), fld)
    end
    return nothing
end

function saveoutputs!(file, model, outputs)
    i = model.clock.iteration
    file["timeseries/t/$i"] = model.clock.time
    for (o, f) in outputs
        file["timeseries/$o/$i"] = f(model)
    end
    return nothing
end

noinit(args...) = nothing

function JLD2OutputWriter(model, outputs; dir=".", prefix="", frequency=1, init=noinit, force=false)
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
    return JLD2OutputWriter(filepath, outputs, frequency)
end

function Oceananigans.write_output(model, fw::JLD2OutputWriter)
    @info @sprintf("Writing JLD2 output %s to %s...", keys(fw.outputs), fw.filepath)
    @time begin
        jldopen(fw.filepath, "r+") do file
            saveoutputs!(file, model, fw.outputs)
        end
    end
    return nothing
end

