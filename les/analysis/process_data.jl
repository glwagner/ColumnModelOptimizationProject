using Glob, JLD2

cd("les")

filepaths = glob("data/*_profiles.jld2")
filepath = filepaths[1]

jldopen(filepath) do file
    @show file
end

cP = 3993.0

function standardize_profile_output(filepath)

    jldopen(filepath, "r+") do file

        for p in keys(file["eos"])
            file["constants/$p"] = file["eos/$p"]
        end

        file["constants/α"] = file["constants/βT"]
        file["constants/cP"] = cP

        file["grid/N"] = file["grid/Nz"]
        file["grid/L"] = file["grid/Lz"]

        for p in keys(file["bcs"])
            file["boundary_conditions/$p"] = file["bcs/$p"]
        end

        α = file["constants/α"]
        g = file["constants/g"]
        file["boundary_conditions/Bz"] = file["bcs/dTdz"] * g * α

        file["constants/κ"] = file["closure/κ"]
        file["constants/ν"] = file["closure/ν"]
    end

    return nothing
end

standardize_profile_output(filepath)
