#=
REALLY IMPORTANT NOTE:
The first method of the report function receives from sciml_train the gradients
of the parameters ∇θ. This functionality was enabled by modifying the source
code of sciml_train. In the file ~/.julia/packages/GalacticOptim/bEh06/src/solve/flux.jl,
I change the line 45 and 65 from "cb_call = cb(θ, x...)" to
"cb_call = cb(θ, G, x...)"; where G is the aforementionated gradient.
=#


using Dates

"""
A structure (variable type) to store the training progress

stg = storage(iter::Int, minimum, minimizer, patience::Int)

FIELDS:
iter:      iteration counter
minimum:   minimum loss found
minimizer: minimum's parameters
patience:  iterations left to find a new minima
"""
mutable struct storage
    iter::Int         # iteration counter
    minimum           # minimum loss found
    minimizer         # minima's parameters
    patience::Int     # iterations left to find a new minima
end
stg = storage(0, Inf, [], 200)    # storage variable




"""
this function reports the training progress and determines if training must halt

halt = report(θ, ∇θ, loss, store_dir)

ARGS:
θ:         current model parameters
∇θ:        parameters' gradient
loss:      parameters' loss
store_dir: directory to store the model parameters

RETURN:
halt: a boolean indicating whether the training must halt
"""
function report(θ, ∇θ, loss, store_dir)
    # account the current iteration and compute the norm of the gradient
    stg.iter += 1
    norm_∇θ  = sqrt(sum(abs2, ∇θ))


    # the callback asks to halt the training when the patience ends, the loss
    # is (practically) zero, or a minimum is reached
    stop_training = (stg.patience == 0) || (loss < 1e-4) || (norm_∇θ < 1e-4)

    # every 50 iterations, the loss and the norm of the gradient are reported
    if  stg.iter%50 == 0 || stg.iter == 1 || stop_training
        println("Iter N° $(stg.iter):
                 [$(now())]
                 loss: $(round(loss, digits = 5))
                 norm(∇θ): $(round(norm_∇θ, digits = 5))")
    end


    # every 5 iterations, the current parameters are stored in disk
    if (stg.iter%5 == 0) || stop_training
        filename = "storage/$(store_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
                   "_iter_$(stg.iter).txt"
        writedlm(filename, θ)
    end


    if stg.minimum < loss
        # the patience is reduced if the current loss is not the new minimum,
        stg.patience -= 1
    else
        # when it is, the parameters are stored and the patience is resettled
        stg.minimum   = loss
        stg.minimizer = θ[:]
        stg.patience  = 2000
    end


    return stop_training
end



"""
this function reports the training progress and determines if training must halt

halt = report(θ, loss, store_dir)

ARGS:
θ:         current model parameters
loss:      parameters' loss
store_dir: directory to store the model parameters

RETURN:
halt: a boolean indicating whether the training must halt
"""
function report(θ, loss, store_dir)
    # account the current iteration and compute the norm of the gradient
    stg.iter += 1


    # the callback asks to halt the training when the patience ends, the loss
    # is (practically) zero, or a minimum is reached
    stop_training = (stg.patience == 0) || (loss < 1e-4)

    # every 50 iterations, the loss and the norm of the gradient are reported
    if  stg.iter%50 == 0 || stg.iter == 1 || stop_training
        println("Iter N° $(stg.iter):
                 [$(now())]
                 loss: $(round(loss, digits = 5))")
    end


    # every 5 iterations, the current parameters are stored in disk
    if (stg.iter%5 == 0) || stop_training
        filename = "storage/$(store_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
                   "_iter_$(stg.iter).txt"
        writedlm(filename, θ)
    end


    if stg.minimum < loss
        # the patience is reduced if the current loss is not the new minimum,
        stg.patience -= 1
    else
        # when it is, the parameters are stored and the patience is resettled
        stg.minimum   = loss
        stg.minimizer = θ[:]
        stg.patience  = 2000
    end


    return stop_training
end