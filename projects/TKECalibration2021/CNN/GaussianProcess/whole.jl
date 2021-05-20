"""
    GaussianProcesses
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""

using Flux

include("./kernels.jl")
include("./distances.jl")
include("./gaussian_processes.jl")

@load "CNN/training_pairs.jld2" xtrain ytrain
@load "CNN/testing_pairs.jld2" xtest ytest

function get_gp_model(xtrain, ytrain, kernel)

    # xtrain .+= rand(Normal(0,1e-10), size(xtrain))

    n = size(xtrain)[2]
    x_train = [xtrain[:,i] for i in 1:n]

    # println("hi", kernel_function(kernel; z = collect(1:length(x_train[1])))(x_train[1],x_train[200]))

    # println(x_train[1])
    # println(ytrain[1])

    models = [GPmodel(x_train, ytrain[i,:]; kernel) for i in 1:6]

    f(x) = [model_output(x, models[i])[1] for i in 1:6]
    return f
end

gp_model = get_gp_model(xtrain, ytrain, SquaredExponentialI(10^(-20.0), 1.0, euclidean_distance))

xs = [xtest[:,i] for i in 1:size(ytest)[2]]
ys = [gp_model(x)[4] for x in xs]
ys_truth = [ytest[4,i] for i in 1:size(ytest)[2]]

scatter([x[1] for x in ys], ys_truth, legend=false)



argmax(ys_truth)
ys_truth[63] = 0.0

using Flux: mse
xs = [xtest[:,i] for i in 1:size(xtest)[2]]
ls = Dict()
for lengthscale = -21.0: 1.0: 25.0
    gp_model = get_gp_model(xtrain, ytrain, Matern12I(10^(-lengthscale), 1.0, euclidean_distance))
    ys = hcat([gp_model(x) for x in xs]...)
    loss = mse(ys, ytest)
    println("loss ", loss)
    ls[lengthscale] = loss
end

ls[7.0]
plot(ls, yscale=:log10, legend=false, color=:purple, lw=4)

argmin(ls)

argmin(ls)
ls[20]
