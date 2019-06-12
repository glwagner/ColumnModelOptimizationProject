function get_gridparams(filename)
    file = jldopen(filename)
    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]

    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    Lz = file["grid/Lz"]
    close(file)

    return Nx, Ny, Nz, Lx, Ly, Lz
end

function Grid(filename)
    Nx, Ny, Nz, Lx, Ly, Lz = get_gridparams(filename)
    RegularCartesianGrid((Nx, Ny, Nz), (Lx, Ly, Lz))
end

function XZGrid(filename)
    Nx, Ny, Nz, Lx, Ly, Lz = get_gridparams(filename)
    RegularCartesianGrid((Nx, 1, Nz), (Lx, Ly, Lz))
end

function get_snapshot(filename, fldname, iter)
    file = jldopen(filename)
    fld = file["timeseries/$fldname/$iter"]
    close(file)
    return fld
end

get_profile_snapshot(args...) = dropdims(get_snapshot(args...), dims=(1, 2))

function get_oceanturb_snapshot(filename, fldname, iter)
    Nx, Ny, Nz, Lx, Ly, Lz = get_gridparams(filename)
    ϕreversed = get_profile_snapshot(filename, fldname, iter)

    grid = OceanTurb.UniformGrid(Nz, Lz)

    ϕ = OceanTurb.CellField(grid)
    OceanTurb.set!(ϕ, reverse(ϕreversed))

    return ϕ
end



function get_iters(filename)
    file = jldopen(filename)
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)
    return iters
end
