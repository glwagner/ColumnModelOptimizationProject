using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux.Losses: mse
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA
using JLD2
using Plots

"""
CNN constructor.
"""
function CNN(args)
    # Conv(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])
    nfeatures₁ = 8
    nfeatures₂ = 12
    σ = leakyrelu
    p_cv = args.p_dropout
    p_fc = args.p_dropout*2
    init_ = Flux.glorot_normal # Glorot (Xavier) initializer.

    return Chain(
            # Conv((5,), imgsize[end]=>nfeatures₁, σ=relu, pad=2), # 160 + padding of 2 = 164
            Conv((5,), 1=>nfeatures₁, σ; pad=2, init=init_), # 160 + padding of 2 = 164
            Flux.Dropout(p_cv),
            Conv((5,), nfeatures₁=>nfeatures₂, σ; init=init_), # 160
            Flux.Dropout(p_cv),
            flatten, # 156
            Dense(156*nfeatures₂, 128, σ; initW=init_, initb=init_),
            Flux.Dropout(p_fc),
            Dense(128, 64, σ; initW=init_, initb=init_),
            Flux.Dropout(p_fc),
            Dense(64, 6)
           )
end

"""
Harvests data from `training_pairs.jld2`.
"""
function get_data(args)

    @load "CNN/training_pairs.jld2" xtrain ytrain
    @load "CNN/testing_pairs.jld2" xtest ytest

    ytrain = Float32.(ytrain)
    ytest = Float32.(ytest)

    xtrain = Float32.(reshape(xtrain, 160, 1, :))
    xtest = Float32.(reshape(xtest, 160, 1, :))

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)

    return train_loader, test_loader
end

"""
Evaluates mean loss across all data pairs from `loader`.
"""
function eval_loss_accuracy(loader, model, device, loss; report=false)
    ps = Flux.params(model)
    l = 0f0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        if report && ntot == 0
            println("Predi. ŷ = $ŷ")
            println("Target y = $y")
        end
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4)
end

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)

# arguments for the `train` function
Base.@kwdef mutable struct Args
    η = 5e-5             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 2        # batch size
    epochs = 200         # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = Int(epochs/2)      # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    savepath = "runs/"   # results path
    p_dropout = 0.05
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = CNN(args) |> device
    @info "CNN model: $(num_params(model)) trainable params"

    ps = Flux.params(model)
    loss(ŷ, y) = Flux.mse(ŷ, y)

    opt = ADAM(args.η)
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(opt, WeightDecay(args.λ))
    end

    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    train_losses = zeros(args.epochs)
    test_losses = zeros(args.epochs)
    function report!(epoch, train_losses, test_losses; printout=false)
        train_loss = eval_loss_accuracy(train_loader, model, device, loss)
        test_loss = eval_loss_accuracy(test_loader, model, device, loss)
        if printout
            println("\n Epoch: $epoch   Train: $train_loss   Test: $test_loss \n")
            return nothing
        end
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" train_loss
                @info "test"  test_loss
            end
        end
        if epoch > 0
            train_losses[epoch] = train_loss
            test_losses[epoch] = test_loss
        end
    end

    !ispath(args.savepath) && mkpath(args.savepath)
    modelpath = joinpath(args.savepath, "model.bson")

    ## TRAINING
    @info "Start Training"
    report!(0, train_losses, test_losses)
    @showprogress for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        report!(epoch, train_losses, test_losses)
        epoch % args.checktime == 0 && report!(epoch, train_losses, test_losses; printout=true)
        if args.checktime > 0 && epoch % args.checktime == 0
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch train_losses test_losses
            end
        end
    end

    println("Model saved in \"$(modelpath)\" \n")

    p = Plots.plot(title="Loss vs. Training Epochs", xlabel="Epochs", ylabel="MSE Loss", size=(350,350))
    Plots.plot!(1:args.epochs, train_losses; label="Training loss", lw=1)
    Plots.plot!(1:args.epochs, test_losses; label="Validation loss", lw=1)
    Plots.savefig("CNN/loss_decay.pdf")

    eval_loss_accuracy(test_loader, model, device, loss; report=true)
    return test_losses[end]
end

## Seach for 🌎 optimum in hyperparameter space

# (1) Learning rate optimization
ηs = 10 .^ (collect(-5:0.5:-3))
ls = zeros(length(ηs))
for (i, η) in enumerate(ηs)
    ls[i] = train(; η=η)
end
p = Plots.plot(ηs, ls, title="Loss vs. Learning rate", xlabel="Learning rate η", ylabel="MSE Validation Loss", size=(350,350), xscale=:log10)
Plots.savefig("CNN/learning_rate.pdf")

η_optimal = ηs[argmin(ls)]

# (2) Batch size optimization
ηs = 10 .^ (collect(-5:0.5:-3))
ls = zeros(length(ηs))
for (i, η) in enumerate(ηs)
    ls[i] = train(; η=η_optimal, )
end
p = Plots.plot(ηs, ls, title="Loss vs. Learning rate", xlabel="Learning rate η", ylabel="MSE Validation Loss", size=(350,350), xscale=:log10)
Plots.savefig("CNN/learning_rate.pdf")

# (3) Dropout amount
