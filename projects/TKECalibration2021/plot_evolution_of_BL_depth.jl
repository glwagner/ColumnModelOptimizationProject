## Plotting mixed layer depth

using TKECalibration2021
LESdata = FourDaySuite # Calibration set
RelevantParameters = TKEFreeConvectionRiIndependent
ParametersToOptimize = TKEFreeConvectionRiIndependent

directory = pwd() * "/TKECalibration2021Results/mixed_layer_depth_vs_time/$(RelevantParameters)/"
isdir(directory) || mkpath(directory)


# les analysis
# derivative function
function δ(z, Φ)
    m = length(Φ)-1
    Φz = ones(m)
    for i in 1:m
        Φz[i] = (Φ[i+1]-Φ[i])/(z[i+1]-z[i])
    end
    return Φz
end

function get_h2(model_time_series, cdata; coarse_grain_data = true)

        # model.model.constants.g * model.model.constants.α

        Qᵇ = 0.001962 * cdata.boundary_conditions.Qᶿ # les.α * les.g * les.top_T
        N² = 0.001962 * cdata.boundary_conditions.dθdz_bottom # les.α * les.g * dθdz_bottom
        Nt = length(cdata.t)

        if coarse_grain_data
            Nz = model_time_series.T[1].grid.N
            z = model_time_series.T[1].grid.zc
        else
            Nz = cdata.grid.N
            z = cdata.grid.zc
        end

        # For the LES solution
        h2_les = randn(Nt)
        for i in 1:Nt

            if coarse_grain_data
                T = CellField(model_time_series.T[i].grid)
                set!(T, cdata.T[i])
                T = T.data[1:Nz]
            else
                T = cdata.T[i].data[1:Nz] # remove halos
            end

            B = 0.001962 * T
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(end-1)] + z[2:end] )./2
            h2_les[i] = -minimum(zA[bools])
        end

        # For the model solution
        h2_model = randn(Nt)
        z = model_time_series.T[1].grid.zc
        Nz = model_time_series.T[1].grid.N
        for i in 1:Nt
            B = 0.001962 * model_time_series.T[i].data
            B = B[1:Nz]
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(end-1)] + z[2:end] )./2

            if 1 in bools
                h2_model[i] = -minimum(zA[bools])
            else
                h2_model[i] = model_time_series.T[1].grid.H # mixed layer reached the bottom
            end
        end

    return [h2_les, h2_model]
end


for file in files
    mymodel = build_model(file);
    cd = mymodel.cdata
    md = mymodel.model

    ℱ = model_time_series(params, md, cd)
    h2_les, h2_model = get_h2(ℱ, cd; coarse_grain_data = false)

    days = @. cd.t / 86400
    toplot = 3:length(cd.t)
    Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=cd.name, ylabel = "Depth [meters]", xlabel = "days", xlims = (0.1, 8.0) , legend = :topleft)
    p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
    Plots.savefig(p, directory*cd.name*"_mixed_layer_depth.pdf")

    p = visualize_realizations(md, cd, 1:180:length(cd), params)
    PyPlot.savefig(directory*cd.name*".png")
end
