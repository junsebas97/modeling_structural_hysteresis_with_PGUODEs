#=
Implementation of the short model with the physics-guided training

PROGRAMMING:          Juan S. Delgado-Trujillo
REVIEWS AND COMMENTS: Diego A. Alvarez
EXPERIMENTAL DATA:    Daniel A. Bedoya-Ruíz
=#

## libraries, functions and directory:
using DiffEqFlux, DifferentialEquations, Plots, Random


include("calc_model_response.jl")
include("loss_func.jl")
include("report.jl")
include("miscelaneous.jl")
include("read_data.jl")
include("astm_load.jl")



## set the seed for random numbers
#Random.seed!(1234)    # Ferrocement wall - MLP of one hidden layer
#Random.seed!(321)     # Ferrocement wall - MLP of three hidden layers
Random.seed!(4321)    # RPL wall




## DATA:
# define the system to identify, the fraction of the test to consider, and the
# downsampling parameter: the maximum deviation between the original and the
#                         downsampled signals

# Ferrocement:
#frac      = 1.0
#max_dev   = 0.0
#directory = "ferrocement_data/"

# RPL:
frac      = 0.7
max_dev   = 0.35
directory = "rpl_data/"


# load the experimental data: one quasi-static cyclic test and the system
# parameters
example, θ_syst = read_data(directory, frac, max_dev)

m = θ_syst["m"]    # store the mass as a global variable



## MODEL:
# formulate the model [step 3], create the multilayer perceptron [step 2] and
# its input normalizing function [step 1]

# STEP 1 - the input normalizing function
x_t     = example["target"]["x(t)"]      # recorded displacement
x_max   = maximum(x_t)                   # maximum recorded displacement
fr_max  = θ_syst["k"]*x_max              # associated elastic restoring force
ε_max   = x_max*fr_max/m                 # maximum mass-normalized stored energy
norm(u) = u./[x_max, 1, fr_max, ε_max]   # u = [x, sign(v), fr, ε]


# STEP 2 - multilayer perceptron
MLP = FastChain((x, p) -> norm(x),         # input normalization
                FastDense(4, 3, tanh),     # first hidden layer
                FastDense(3, 3, tanh),     # second hidden layer
                FastDense(3, 3, tanh),     # third hidden layer
                FastDense(3, 1))           # output layer



# STEP 3 - short model
"""
Short model: an UODE with the equation of motion, the dynamics of the
restoring force, and the dynamics mass normalized dissipated energy. Here a
multilayer perceptron models the restoring force.

Equations and units:
1) dx/dt      = v(t)
   [mm/s]     = [mm/s]
2) dv/dt      = [p(t) - c*v(t) - fr(t)]/[1e-6*m]
   [mm/s²]    = ([kN] - [kN*s/mm]*[mm/s] - [kN])/[10⁶ kg]
3) dfr/dt     = MLP(.)*v(t)
   [kN/s]     = [kN/mm]*[mm/s]
4) dε/dt      = fr(t)*v(t)/m
   [(J/kg)/s] = [kN]*[mm/s]/[kg] = m²/s³ = J/(kg*s)
"""
function model(du, u, θ, t, p)
    x, v, fr, ε = u                      # state variables
    c = θ[1];        θ_mlp = θ[2:end]    # model parameters

    du[1] = dx  = v
    du[2] = dv  = (p(t) - c*v - fr)/(1e-6*m)
    du[3] = dfr = MLP([x, sign(v), fr, ε], θ_mlp)[1]*v
    du[4] = dε  = fr*v/m
end


# create a directory to store the parameters through the training and pass this
# directory to the report function
model_dir = "results/"
mkpath("storage/"*model_dir)
report(θ, ∇θ, loss) = report(θ, ∇θ, loss, model_dir)



## TRAINING:
#=
The training of the proposed model is performed in three phases, each one with
several rounds because the training tends to stagnate; thus, it needs to be
restarted.

Phase 1: The model is trained using only experimental data

Phase 2: The model is trained using experimental data and some physical
         restrictions. In this phase, most physical restrictions do not apply to
         the model response for the elastic loads and free vibrations.

Phase 3: The model is trained using experimental data and physical restrictions
         on all on the model responses.
=#

# define the analyzed times of the free vibration responses
t_fv = 0:0.05:10    # [s]


# initialize the model parameters
θ_model = good_initialize(example, θ_syst, t_fv)
writedlm("storage/$(model_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
         "_initial_params.txt", θ_model)


# Phase 1:
# define the maximum iterations and the optimizer in each round
n_iter    = [      1000,       1000,       1000]
optimizer = [ADAM(0.01), ADAM(5e-3), ADAM(1e-3)]
n_round   = length(n_iter)

# incorporate the experimental data into the loss function
loss_data(θ) = loss_data(θ, example)

for rd = 1:n_round
    println("Round N° $(rd)")
    # training:
    DiffEqFlux.sciml_train(loss_data, θ_model, optimizer[rd],
                           maxiters = n_iter[rd], cb = report)
    print("\n\n*****End round*****\n\n")
    
    # after each round, the minimizer is stored and assigned to be the new model
    # parameters and the storage variable is restarted
    writedlm("storage/$(model_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
             "_phase1_end_round$(rd).txt", stg.minimizer)
    global θ_model = stg.minimizer
    global stg     = storage(0, stg.minimum, stg.minimizer, 2000)
end

# initialize the storage variable for the next phase
stg = storage(0, Inf, [], 2000)


# Phase 2:
# define the maximum iterations and the optimizer in each round
n_iter    = [       500,       1000,       1500,       2000,       2500]
optimizer = [ADAM(0.01), ADAM(0.01), ADAM(5e-3), ADAM(1e-3), ADAM(5e-4)]
n_round   = length(n_iter)

# incorporate the experimental data into the loss function
loss_physics(θ) = loss_physics(θ, example, θ_syst, t_fv)

for rd = 1:n_round
    println("Round N° $(rd)")
    # training:
    DiffEqFlux.sciml_train(loss_physics, θ_model, optimizer[rd],
                           maxiters = n_iter[rd], cb = report)
    print("\n\n*****End round*****\n\n")
    
    # after each round, the minimizer is stored and assigned to be the new model
    # parameters and the storage variable is restarted
    writedlm("storage/$(model_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
             "_phase2_end_round$(rd).txt", stg.minimizer)
    global θ_model = stg.minimizer
    global stg     = storage(0, stg.minimum, stg.minimizer, 2000)
end

# initialize the storage variable for the next phase
stg = storage(0, Inf, [], 2000)


# Phase 3:
# define the maximum iterations and the optimizer in each round
n_iter    = [       500,        500,       1000,       1000,       1500]
optimizer = [ADAM(1e-4), ADAM(1e-4), ADAM(5e-5), ADAM(1e-5), ADAM(1e-5)]
n_round   = length(n_iter)

# incorporate the experimental data into the loss function
loss_extended(θ) = loss_extended(θ, example, θ_syst, t_fv)

for rd = 1:n_round
    println("Round N° $(rd)")
    # training:
    DiffEqFlux.sciml_train(loss_extended, θ_model, optimizer[rd],
                           maxiters = n_iter[rd], cb = report)
    print("\n\n*****End round*****\n\n")
    
    # after each round, the minimizer is stored and assigned to be the new model
    # parameters and the storage variable is restarted
    writedlm("storage/$(model_dir)/$(Dates.format(now(), "yy_mm_dd_HH_MM_SS"))"*
             "_phase3_end_round$(rd).txt", stg.minimizer)
    global θ_model = stg.minimizer
    global stg     = storage(0, stg.minimum, stg.minimizer, 2000)
end



## VALIDATION:
# to validate the model, we assess its response for the load pattern 1 of the
# test A of ASTM 2126. Furthermore, we plot the model response and detail the
# loss for all excitation cases used in training:

# define the maximum magnitude of the validation load:
t_i   = example["input"]["t"]       # times of experimental data
p_fun = example["input"]["p(t)"]    # p(t) experimental data  --->  function
p_i   = p_fun.(t_i)                 #                         --->  times series
Fv    = 0.75*maximum(p_i)

print_validation_results(θ_model, example, θ_syst, t_fv, Fv)