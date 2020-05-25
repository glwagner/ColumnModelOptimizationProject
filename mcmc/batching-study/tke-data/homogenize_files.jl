using ColumnModelOptimizationProject, Glob

include("../../setup.jl")
include("../../utils.jl")

filenames = glob("*scaled*.jld2")
@show filenames

for filename in filenames
    file = jldopen(filename)
    calibration_name = keys(file)
    calibration = file[calibration_name[1]] # assumes calibration name is only group
    close(file)

    @show filename calibration_name

    # Savely save a new file.
    tempfilename = filename[1:end-5] * "-temp.jld2"
    mv(filename, tempfilename)

    @save filename calibration # changing all instances to simply "calibration"

    rm(tempfilename)
end
