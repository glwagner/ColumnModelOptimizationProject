function get_iterations(datapath)
    file = jldopen(datapath, "r")
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)
    return iters
end

function get_times(datapath)
    iters = iterations(datapath)
    t = zeros(length(iters))
    jldopen(datapath, "r") do file
        for (i, iter) in enumerate(iters)
            t[i] = file["timeseries/t/$iter"]
        end
    end
    return t
end

function get_parameter(filename, group, parameter_name)
    parameter = nothing

    jldopen(filename) do file
        if parameter_name ∈ keys(file["$group"])
            parameter = file["$group/$parameter_name"]
        end
    end

    return parameter
end

function get_data(varname, datapath, iter; reversed=false)
    file = jldopen(datapath, "r")
    var = file["timeseries/$varname/$iter"]
    close(file)

    # Drop extra singleton dimensions if they exist
    if ndims(var) > 1
        droplist = []
        for d = 1:ndims(var)
           size(var, d) == 1 && push!(droplist, d)
       end
       var = dropdims(var, dims=Tuple(droplist))
    end

    reversed && reverse!(var)

    return var
end

function get_grid_params(datapath::String)
    file = jldopen(datapath, "r")
    N = file["grid/Nz"]
    L = file["grid/Lz"]
    close(file)
    return N, L
end
