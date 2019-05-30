using Plots, Oceananigans, Statistics, OceananigansAnalysis
    


maxabs(u) = maximum(abs.(u))
Umax(u, v, w) = max(maxabs(u), maxabs(v), maxabs(w))
Umax(model) = Umax(model.velocities.u.data, model.velocities.v.data, model.velocities.w.data)
Δmin(grid) = min(grid.Δx, grid.Δy, grid.Δz)
cfl(Δt, model) = Δt * Umax(model) / Δmin(model.grid)

function safe_Δt(model, αu, αν=0.01)
    τu = Δmin(model.grid) / Umax(model)
    τν = Δmin(model.grid)^2 / model.closure.ν

    return min(αν*τν, αu*τu)
end

xnodes(ϕ) = repeat(reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ) = repeat(reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ) = repeat(reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

xnodes(ϕ::FaceFieldX) = repeat(reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ::FaceFieldY) = repeat(reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ::FaceFieldZ) = repeat(reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

nodes(ϕ) = (xnodes(ϕ), ynodes(ϕ), znodes(ϕ))

zerofunk(args...) = 0

function set_ic!(model; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk)
    data(model.velocities.u) .= u.(nodes(model.velocities.u)...)
    data(model.velocities.v) .= v.(nodes(model.velocities.v)...)
    data(model.velocities.w) .= w.(nodes(model.velocities.w)...)
    data(model.tracers.T) .= T.(nodes(model.tracers.T)...)
    data(model.tracers.S) .= S.(nodes(model.tracers.S)...)
    return nothing
end

plot_xzslice(ϕ, slice=1, args...; kwargs...) = 
    pcolormesh(view(xnodes(ϕ), :, slice, :), view(znodes(ϕ), :, slice, :), 
               view(data(ϕ), :, slice, :), args...; kwargs...)

plot_xyslice(ϕ, slice=1, args...; kwargs...) = 
    pcolormesh(view(xnodes(ϕ), :, :, slice), view(ynodes(ϕ), :, :, slice), 
               view(data(ϕ), :, :, slice), args...; kwargs...)

function plot_hmean(ϕ, args...; normalize=false, kwargs...)
    ϕhmean = dropdims(mean(data(ϕ), dims=(1, 2)), dims=(1, 2))
    if !normalize
        ϕnorm = 1
    else
        ϕmean = mean(data(ϕ))
        ϕhmean = ϕhmean .- ϕmean
        ϕnorm = maximum(ϕhmean) - minimum(ϕhmean)
    end
    PyPlot.plot(ϕhmean/ϕnorm, ϕ.grid.zC, args...; kwargs...)
end


total_kinetic_energy(u, v, w) = 
    0.5 * (sum(u.data.^2) + sum(v.data.^2) + sum(w.data.^2))

total_kinetic_energy(model) = total_kinetic_energy(model.velocities...)

function total_energy(model)
    b = model.tracers.T.data .- mean(model.tracers.T.data, dims=(1, 2))
    return total_kinetic_energy(model.velocities...) + 0.5 * sum(b.^2) / N^2
end

function buoyancy_flux(model::Model{A}) where A
    Nx = model.grid.Nx
    Ny = model.grid.Ny
    Nz = model.grid.Nz
    w = model.velocities.w
    T = model.tracers.T
    g, βT = model.constants.g, model.eos.βT

    wb = CellField(A(), model.grid)
    @views @. wb.data[1:Nx, 1:Ny, 1:Nz-1] = (
        g * βT * 0.5 * (w[1:Nx, 1:Ny, 1:Nz-1] + w[1:Nx, 1:Ny, 2:Nz]) * T[1:Nx, 1:Ny, 1:Nz-1])

    # Assumes a no-penetration boundary condition
    @views @. wb.data[1:Nx, 1:Ny, Nz] = g * βT * 0.5 * w[1:Nx, 1:Ny, Nz] * T[1:Nx, 1:Ny, Nz]

    return wb
end

function fluctuation_variance(ϕ)
    ϕ² = CellField(CPU(), ϕ.grid)
    @. ϕ².data = (ϕ.data - $mean($data(ϕ), dims=(1, 2)))^2
    return ϕ²
end

function kinetic_energy(model::Model{A}) where A
    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w

    e = CellField(A(), model.grid)

    kinetic_energy!(e, u, v, w)

    return e
end

function kinetic_energy!(e, u, v, w)
    Nx, Ny, Nz = e.grid.Nx, e.grid.Ny, e.grid.Nz

    @views @. e.data[1:Nx, 1:Ny, 1:Nz-1] = (0.25 *
                (  u[1:Nx, 1:Ny, 1:Nz-1]^2 + u[2:Nx+1, 1:Ny,   1:Nz-1]^2
                 + v[1:Nx, 1:Ny, 1:Nz-1]^2 + v[1:Nx,   2:Ny+1, 1:Nz-1]^2
                 + w[1:Nx, 1:Ny, 1:Nz-1]^2 + w[1:Nx,   1:Ny,   2:Nz]^2
                ))

    @views @. e.data[1:Nx, 1:Ny, Nz] = (0.25 *
                (  u[1:Nx, 1:Ny, Nz]^2 + u[2:Nx+1, 1:Ny,   Nz]^2
                 + v[1:Nx, 1:Ny, Nz]^2 + v[1:Nx,   2:Ny+1, Nz]^2
                 + w[1:Nx, 1:Ny, Nz]^2
                ))

    return e
end

function turbulent_kinetic_energy(model::Model{A}) where A
    u′, v′, w′ = fluctuations(model.velocities.u, model.velocities.v, model.velocities.w)
    e = CellField(A(), model.grid)
    kinetic_energy!(e, u′, v′, w′)
    return e
end

"""
    make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                              var_name, Nt, Δt, var_offset=0, slice_idx=1)

Make a movie of a vertical slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀). A `slice_idx` can be specified to select
the index of the y-slice to be plotted (useful when plotting vertical slices
from a 3D model, it should be set to 1 for 2D xz-slice models).
"""
function make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0, slice_idx=1)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.contour(model.grid.xC, reverse(model.grid.zC), rotl90(var[:, slice_idx, :] .- var_offset),
                      fill=true, levels=9, linewidth=0, color=:balance,
                      clims=(-0.011, 0.011), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
        # Plots.heatmap(model.grid.xC, model.grid.zC, rotl90(var[:, slice_idx, :]) .- var_offset,
        #               color=:balance, clims=(-0.01, 0.01), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a horizontal slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.heatmap(model.grid.xC, model.grid.yC, var[:, :, 1] .- var_offset,
                      color=:balance, clims=(-0.01, 0.01),
                      title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a vertical profile produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.plot(var[1, 1, :] .- var_offset, model.grid.zC,
                   title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

using NetCDF

function make_avg_temperature_profile_movie()
    Nt, dt = 86400, 0.5
    freq = 3600
    N_frames = Int(Nt/freq)
    filename_prefix = "convection"
    var_offset = 273.15

    Nz, Lz = 128, 100
    dz = Lz/Nz
    zC = -dz/2:-dz:-Lz

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")

        filepath = filename_prefix * lpad(freq*n, 9, "0") * ".nc"
        field_data = ncread(filepath, "T")
        ncclose(filepath)

        T_profile = mean(field_data; dims=[1,2])

        Plots.plot(reshape(T_profile, Nz) .- var_offset, zC,
                   title="t=$(freq*n*dt) s ($(round(freq*n*dt/86400; digits=2)) days)")
    end

    mp4(animation, filename_prefix * "$(round(Int, time())).mp4", fps=30)
end
