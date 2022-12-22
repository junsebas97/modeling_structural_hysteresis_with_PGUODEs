using Interpolations

"""
Linear interpolation

function lin_interp(x, x1y1, x2y2)

ARGS:
x:
x1y1:   coordinates of point 1
x2y2:   coordinates of point 2

RETURNS:
y:
"""
function lin_interp(x, x1y1, x2y2)
    (x1, y1) = x1y1
    (x2, y2) = x2y2

    m = (y2-y1)/(x2-x1)
    b = (x2*y1 - x1*y2)/(x2-x1)
    
    return m*x + b
end



## DATA-DRIVEN LOSS:
"""
Data-driven loss function

loss = loss_data(θ, example)

ARGS:
θ:       model parameters
example: training example contaning the input, target, and envelopes, where:
            - the input has the time "t", the external force "p(t)", and the 
            initial conditions "u0",
            - the target corresponds to recorded displacements "x(t)" and
            computed dissipated energy "ε(t)", and
            - the envelopes are the displacement "Sx(t)" and dissipated energy
            "Sε(t)" envelopes.

RETURNS: loss over the experimental data
"""
function loss_data(θ, example)
    # extract the example data
    t  = example["input"]["t"]
    p  = example["input"]["p(t)"]
    u0 = example["input"]["u0"]

    x_true = example["target"]["x(t)"]
    ε_true = example["target"]["ε(t)"]

    Sx = example["envelopes"]["Sx(t)"]
    Sε = example["envelopes"]["Sε(t)"]


    n_dat = length(t) - 1


    # solve the proposed model for the excitation of the experimental data
    model_resp = calc_model_response(p, u0, θ, t)
    x_pred     = model_resp[:, 1]
    ε_pred     = model_resp[:, 4]


    # if the model does not diverge compute the loss:
    if length(x_pred) == (n_dat + 1)

        # calculate the error on the displacement and dissipated energy, assess
        # the loss weights, and compute the data-driven loss
        loss_x = sum(abs2, (x_true[2:end] .- x_pred[2:end])./Sx[2:end])/n_dat
        loss_ε = sum(abs2, (ε_true[2:end] .- ε_pred[2:end])./Sε[2:end])/n_dat

        loss = loss_x + loss_ε

    # otherwise, return an infinity loss value
    else
        loss = Inf
    end

    # return the loss value
    return loss
end





## PHYSICS-GUIDED LOSS FUNCTIONS:

# Constraints:
"""
Constraint of asymptotic dissipativity in free vibration

loss = asym_dissip(x, v, k, m)

ARGS:
x: displacement in a free vibration case    [mm]
v: velocities in a free vibration case      [mm/s]
k: initial stiffness                        [kN/mm]
m: mass                                     [kg]

RETURNS: penalty
"""
function asym_dissip(x, v, k, m)
    # calculate the stored energy in the system (equation 8 of
    # Ikhouane and Rodellar (2005)
    # Physical consistency of the hysteretic Bouc-Wen model
    # https://doi.org/10.3182/20050703-6-CZ-1902.00147
    Es    = (1/2)*(1e-6*m)*v.^2 + (1/2)*k*x.^2
    # [J] =      [10⁶ kg][mm/s]² + [kN/mm][mm]²

    # and its change
    ΔEs = diff(Es)

    # compute the loss by summing energy gains; that is, positive energy changes
    return sum(ΔEs[ ΔEs .> 0 ])
end



"""
Constraint of asymptotic motion in free vibration

loss = asymptotic_motion(v, frac)

ARGS:
v:    velocity in a free vibration case      [mm/s]
frac: proportion of end data to be analyzed

RETURNS: penalty
"""
function asymptotic_motion(v, frac=0.95)
    n_dat = length(v)

    # analyzed interval, it is far from the beggining of the motion
    idx = round(Int, frac*n_dat):n_dat

    # in this interval, sum all the nonzero velocities
    return sum(abs, v[idx])
end



"""
Constraint of Drucker postulate of plasticity

loss = Drucker(fr, ε)

ARGS:
fr: restoring force
ε:  dissipated energy

RETURNS: penalty
"""
function Drucker(fr, ε)
    n_dat = size(fr, 1)
    loss  = 0.0

    # in each point:
    for i = 1:n_dat
        ε_fl = 0.0    # dissipated energy in the forward loop
        ε_bl = 0.0    # dissipated energy in the backward loop

        # 1) find the next point with equal restoring force, that is, the end
        # of the forward closed load loop
        for j = (i + 1):(n_dat - 1)

            # when the system response is constant, such point is in the series
            if fr[j] == fr[i] == fr[j + 1]

                ε_fl = ε[j] - ε[i]
                break

            # otherwise, we must estimate it
            elseif (fr[j] < fr[i] <= fr[j + 1]) || (fr[j] > fr[i] >= fr[j + 1])

                # 1A) compute dissipated energy in this point
                εj = lin_interp(fr[i], (fr[j], ε[j]), (fr[j + 1], ε[j + 1]))

                # 1B) assess the loop's work in the forward closed load loop
                ε_fl = εj - ε[i]
                break
            end
        end


        # 2) find the previous point with equal restoring force, that is, the
        # beginning of the backward closed load loop
        for k = (i - 1):-1:2

            # when the system response is constant, such point is in the series
            if fr[k] == fr[i] == fr[k + 1]

                ε_bl = ε[i] - ε[k]
                break

            # otherwise, we must estimate it
            elseif (fr[k] < fr[i] <= fr[k - 1]) || (fr[k] > fr[i] >= fr[k - 1])

                # 2A) compute dissipated energy in this point
                εk = lin_interp(fr[i], (fr[k], ε[k]), (fr[k - 1], ε[k - 1]))

                # 2B) assess the loop's work in the backward closed load loop
                ε_bl = ε[i] - εk
                break
            end
        end


        # 3) if any of these works is negative, add it to the loss
        loss += max(0.0, -ε_fl) + max(0.0, -ε_bl)
    end

    return loss
end



"""
Constraint of free vibration displacement of linear systems

loss = free_vib_displ(θ, x_pred, u0, k, m, t)

ARGS:
θ:      model parameters
x_pred: model displacements for a free vibration excitation      [mm]
u0:     initial displacement and velocity of free vibration      [mm], [mm/s]
k:      initial stiffness of the system                          [kN/mm]
m:      system mass                                              [kg]
t:      analyzed times                                           [s]

RETURNS: penalty
"""
function free_vib_displ(θ, x_pred, u0, k, m, t)
    n_dat = length(t) - 1

    # take the damping identified by the model and the initial conditions
    c        = θ[1]
    (x0, v0) = u0


    # assess the natural frequency ω0, damping ratio ξ, and damped frequency ωD
    ω0 = sqrt(1e6*k/m)       # [1/s] = sqrt(10e-6*[kN/mm]/[kg])

    ξ  = 1e6*c/(2*m*ω0)      # [-]   = 10e-6*[kN-s/mm]/([kg][1/s])
    ξ  = min(ξ, 0.999)       # ξ<1 in order to avoid complex solutions

    ωD = ω0*sqrt(1 - ξ^2)    # [1/s] = [1/s][-]
    

    # calculate the true displacement for the free vibration (EQ. 2-11) and
    # its envelope
    x_true = (x0*cos.(ωD*t) + ((v0 + x0*ξ*ω0)/ωD)*sin.(ωD*t)).*exp.(-ξ*ω0*t)
    Sx     = 1 .+ exp.(-ξ*ω0*t)


    # compute and return the modulated MSE of the model response for free
    # vibration:
    return (1/n_dat)*sum(abs2, (x_true[2:end] .- x_pred[2:end])./Sx[2:end])
end




"""
constraint of initial elasticity

loss = initial_elasticity(x, uy)

ARGS:
x:  displacements for an elastic loading    [mm]
uy: yielding displacement of the system     [mm]

RETURNS: penalty
"""
function initial_elasticity(x, uy)
    # take the last displacement and compute the residual as a fraction of the
    # yielding displacement
    xr = x[end]/uy
    
    # compute the loss by returning the residual
    return abs(xr)
end



"""
first constraint of the BIBO stability

loss = first_BIBO(x, fr)

ARGS:
x:  displacements         [mm]
fr: restoring forces      [kN]

RETURNS: penalty
"""
function first_BIBO(x, fr)
    n_dat  = length(fr)
    x_mag  = abs.(x)        # displacement magnitude
    fr_mag = abs.(fr)       # magnitude of the restoring force
    loss   = 0

    # in the displacement reversals:
    for i = 2:(n_dat - 1)
        if x_mag[i - 1] < x_mag[i] > x_mag[i + 1]
            # compute the change in the restoring force magnitude,
            ΔFr = fr_mag[i + 1] - fr_mag[i]

            # and add it to the loss if it is possitive
            loss += max(0.0, ΔFr)
        end
    end
    
    return loss
end



"""
Constraint of the Iliushin postulate of plasticity

loss = Iliushyn(x, ε)

ARGS:
x: displacements
ε: dissipated energy

RETURNS: penalty
"""
function Iliushyn(x, ε)
    n_dat = size(x, 1)
    loss  = 0.0

    # in each point:
    for i = 1:n_dat
        ε_fl = 0.0    # dissipated energy in the forward loop
        ε_bl = 0.0    # dissipated energy in the backward loop

        # 1) find the next point with the same displacement, that is, the end
        # of the forward closed displacement loop
        for j = (i + 1):(n_dat - 1)

            # when the system response is constant, such point is in the series
            if x[j] == x[i] == x[j + 1]

                ε_fl = ε[j] - ε[i]
                break
            
            # otherwise, we must estimate it
            elseif (x[j] < x[i] <= x[j + 1]) || (x[j] > x[i] >= x[j + 1])

                # 1A) compute the dissipated energy in this point
                εj = lin_interp(x[i], (x[j], ε[j]), (x[j + 1], ε[j + 1]))

                # 1B) assess the loop's work in the forward closed displacement
                # loop
                ε_fl = εj - ε[i]
                break
            end
        end


        # 2) find the previous point with the same displacement, that is, the
        # beginning of the backward closed displacement loop
        for k = (i - 1):-1:2

            # when the system response is constant, such point is in the series
            if x[k] == x[i] == x[k - 1]

                ε_bl = ε[i] - ε[k]
                break

            # otherwise, we must estimate it
            elseif (x[k] < x[i] <= x[k - 1]) || (x[k] > x[i] >= x[k - 1])

                # 2A) compute the dissipated energy in this point
                εk = lin_interp(x[i], (x[k], ε[k]), (x[k - 1], ε[k - 1]))

                # 2B) assess the loop's work in the backward closed displacement
                # loop
                ε_bl = ε[i] - εk
                break
            end
        end


        # 3) if any of these works is negative, add it to the loss
        loss += max(0.0, -ε_fl) + max(0.0, -ε_bl)
    end

    return loss
end



"""
Constraint of the passivity property

loss = passivity(ε)

ARGS:
ε: dissipated energy

RETURNS: penalty
"""
function passivity(ε)
    # Extract and sum negative dissipated energies to compute the loss
    return -sum(ε[ε .< 0])
end



"""
Constraint of the property of possitive damping

loss = positive_damping(θ)

ARGS:
θ: current paramaters of the model

RETURNS: penalty
"""
function positive_damping(θ)
    # Extract the first parameter (damping) and return it if negative
    return max(0.0, -θ[1])
end



"""
second constraint of the BIBO stability

loss = second_BIBO(k, x, fr, α=10)

ARGS:
k:  initial stiffness
x:  displacements
fr: restoring forces
α:  factor that increases the stiffness

RETURNS: penalty
"""
function second_BIBO(k, x, fr, α=10)
    # assess largest displacement and define the restoring force limit
    xL_max = maximum(abs.(x))
    L_fr   = α*k*xL_max

    # calculate the magnitude of the restoring forces
    fr_mag = abs.(fr)

    # compute the loss by summing all the restoring forces that exceed the limit
    return sum(fr_mag[fr_mag .> L_fr])
end





# Loss functions:

"""
loss function with a data-driven and a physics component

loss = loss_physics(θ, example, θ_system, t_fv)

ARGS:
θ:       model parameters
example: training example contaning the input, target, and envelopes, where:
           - the input has the time "t", the external force "p(t)", and the
             initial conditions "u0",
           - the target corresponds to recorded displacements "x(t)" and
             computed dissipated energy "ε(t)", and
           - the envelopes are the displacement "Sx(t)" and dissipated energy
             "Sε(t)" envelopes.
θ_syst:  system parameters, that is, the mass "m", initial stiffness "k", and
         yielding displacement "uy".
t_fv:    times of the free vibration responses.

RETURNS: loss over the experimental data and the governing physics
"""
function loss_physics(θ, example, θ_system, t_fv)
    # take the system parameters
    m  = θ_system["m"]
    k  = θ_system["k"]
    uy = θ_system["uy"]


    # extract the experimental data
    t  = example["input"]["t"]
    p  = example["input"]["p(t)"]
    u0 = example["input"]["u0"]

    x_true = example["target"]["x(t)"]
    ε_true = example["target"]["ε(t)"]

    Sx = example["envelopes"]["Sx(t)"]
    Sε = example["envelopes"]["Sε(t)"]

    n_dat = length(t) - 1


    # define the the free vibration cases:
    (x0, v0) = (0.95*uy, 1.0)

    p_fv(t) = 0.0
    u0_ed   = [ x0, 0.0, k*x0, k*(x0^2)/2]    # [x0  v0  fr0  ξ0]
    u0_uv   = [0.0,  v0,  0.0,        0.0]    # [x0  v0  fr0  ξ0]


    # define the elastic loadings:
    t_el  = 0.0:0.1:120
    u0_el = [0.0, 0.0, 0.0, 0.0]           # [x0  v0  fr0  ξ0]
    p_1   = LinearInterpolation([0., 30, 60, 120], [0, 0.25*k*uy, 0, 0])
    p_2   = LinearInterpolation([0., 30, 60, 120], [0, 0.50*k*uy, 0, 0])
    p_3   = LinearInterpolation([0., 30, 60, 120], [0,      k*uy, 0, 0])



    # solve the model for all the above excitation cases:
    u    = calc_model_response(   p,     u0, θ,    t)
    u_ed = calc_model_response(p_fv,  u0_ed, θ, t_fv)
    u_uv = calc_model_response(p_fv,  u0_uv, θ, t_fv)
    u_1  = calc_model_response( p_1,  u0_el, θ, t_el)
    u_2  = calc_model_response( p_2,  u0_el, θ, t_el)
    u_3  = calc_model_response( p_3,  u0_el, θ, t_el)



    # extract the variables of each model response needed by the loss function:
    # 1) the excitation of the experimental data:
    x = u[:, 1];          fr = u[:, 3];       ε = u[:, 4]
    # 2) the free vibration cases:
    x_ed = u_ed[:, 1];    v_ed = u_ed[:, 2]
    x_uv = u_uv[:, 1];    v_uv = u_uv[:, 2]
    # 3) three elastic loads:
    x_1 = u_1[:, 1];      x_2 = u_2[:, 1];    x_3 = u_3[:, 1]



    # if the model does not diverges in the excitation case of experimental data
    # and the free vibrations compute the loss:
    if length(x)    == (n_dat + 1)  &&
       length(x_ed) == length(t_fv) &&
       length(x_uv) == length(t_fv)

        # compute the data-driven component of the loss
        loss_x = (1/n_dat)*sum(abs2, (x_true[2:end] .- x[2:end])./Sx[2:end])
        loss_ε = (1/n_dat)*sum(abs2, (ε_true[2:end] .- ε[2:end])./Sε[2:end])

        # compute the physics component of the loss
        C1 = first_BIBO(x, fr) + second_BIBO(k, x, fr)    # C1A + C1B
        C2 = passivity(ε)
        C3 = positive_damping(θ)
        C4 = asymptotic_motion(v_ed) + asymptotic_motion(v_uv)
        C5 = asym_dissip(x_ed, v_ed, k, m) + asym_dissip(x_uv, v_uv, k, m)
        #C6 = free_vib_displ(θ, x_ed, (x0,  0), k, m, t_fv) + 
        #     free_vib_displ(θ, x_uv, ( 0, v0), k, m, t_fv)
        C6 = 0
        C7 = initial_elasticity(x_1, uy) + initial_elasticity(x_2, uy) +
             initial_elasticity(x_3, uy)
        C8 = Drucker(fr, ε)
        C9 = Iliushyn(x, ε)

        # calculate the loss
        loss = loss_x + loss_ε + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9

    # otherwise, return an infinity loss value:
    else
        loss = Inf
    end


    # return the loss value
    return loss
end





"""
physics-guided loss function that extends the constraints of BIBO stability,
passivity, and plasticity postulates to the responses of free vibrations and
elastic loads

loss = loss_extended(θ, example, θ_system, t_fv)

ARGS:
θ:       model parameters
example: training example contaning the input, target, and envelopes, where:
           - the input has the time "t", the external force "p(t)", and the 
             initial conditions "u0",
           - the target corresponds to recorded displacements "x(t)" and
             computed dissipated energy "ε(t)", and
           - the envelopes are the displacement "Sx(t)" and dissipated energy
             "Sε(t)" envelopes.
θ_syst:  system parameters, that is, the mass "m", initial stiffness "k", and
         yielding displacement "uy".
t_fv:    times of the free vibration responses.

RETURNS: loss over the experimental data and the governing physics
"""
function loss_extended(θ, example, θ_system, t_fv)
    # take the system parameters
    m  = θ_system["m"]
    k  = θ_system["k"]
    uy = θ_system["uy"]


    # extract the experimental data
    t  = example["input"]["t"]
    p  = example["input"]["p(t)"]
    u0 = example["input"]["u0"]

    x_true = example["target"]["x(t)"]
    ε_true = example["target"]["ε(t)"]

    Sx = example["envelopes"]["Sx(t)"]
    Sε = example["envelopes"]["Sε(t)"]

    n_dat = length(t) - 1



    # define the the free vibration cases:
    (x0, v0) = (0.95*uy, 1.0)

    p_fv(t) = 0.0
    u0_ed   = [ x0, 0.0, k*x0, k*(x0^2)/2]    # [x0  v0  fr0  ξ0]
    u0_uv   = [0.0,  v0,  0.0,        0.0]    # [x0  v0  fr0  ξ0]




    # define the elastic loadings:
    t_el  = 0.0:0.1:120
    u0_el = [0.0, 0.0, 0.0, 0.0]           # [x0  v0  fr0  ξ0]
    p_1   = LinearInterpolation([0., 30, 60, 120], [0, 0.25*k*uy, 0, 0])
    p_2   = LinearInterpolation([0., 30, 60, 120], [0, 0.50*k*uy, 0, 0])
    p_3   = LinearInterpolation([0., 30, 60, 120], [0,      k*uy, 0, 0])



    # solve the model for all the above excitation cases:
    u    = calc_model_response(   p,    u0, θ,    t)
    u_ed = calc_model_response(p_fv, u0_ed, θ, t_fv)
    u_uv = calc_model_response(p_fv, u0_uv, θ, t_fv)
    u_1  = calc_model_response( p_1, u0_el, θ, t_el)
    u_2  = calc_model_response( p_2, u0_el, θ, t_el)
    u_3  = calc_model_response( p_3, u0_el, θ, t_el)



    # extract the variables for the model response of:
    # 1) the excitation of the experimental data:
    x  = u[:, 1]
    fr = u[:, 3]
    ε  = u[:, 4]
    # 2) the free vibration cases:
    x_ed  = u_ed[:, 1];      x_uv  = u_uv[:, 1]
    v_ed  = u_ed[:, 2];      v_uv  = u_uv[:, 2]
    fr_ed = u_ed[:, 3];      fr_uv = u_uv[:, 3]
    ε_ed  = u_ed[:, 4];      ε_uv  = u_uv[:, 4]
    # 3) three elastic loads:
    x_1  = u_1[:, 1];      x_2  = u_2[:, 1];      x_3  = u_3[:, 1]
    fr_1 = u_1[:, 3];      fr_2 = u_2[:, 3];      fr_3 = u_3[:, 3]
    ε_1  = u_1[:, 4];      ε_2  = u_2[:, 4];      ε_3  = u_3[:, 4]



    # if the model does not diverges in the excitation case of experimental data
    # and the free vibrations compute the loss:
    if length(x)    == (n_dat + 1)  &&
       length(x_ed) == length(t_fv) &&
       length(x_uv) == length(t_fv)

        # compute the data-driven loss and the physical constraints
        loss_x = (1/n_dat)*sum(abs2, (x_true[2:end] .- x[2:end])./Sx[2:end])
        loss_ε = (1/n_dat)*sum(abs2, (ε_true[2:end] .- ε[2:end])./Sε[2:end])

        C1 = first_BIBO(x,       fr) + second_BIBO(k,    x,    fr) +
             first_BIBO(x_ed, fr_ed) + second_BIBO(k, x_ed, fr_ed) +
             first_BIBO(x_uv, fr_uv) + second_BIBO(k, x_uv, fr_uv) +
             first_BIBO(x_1,   fr_1) + second_BIBO(k,  x_1,  fr_1) +
             first_BIBO(x_2,   fr_2) + second_BIBO(k,  x_2,  fr_2) +
             first_BIBO(x_3,   fr_3) + second_BIBO(k,  x_3,  fr_3)

        C2 = passivity(ε)   + passivity(ε_ed) + passivity(ε_uv) +
             passivity(ε_1) + passivity(ε_2)  + passivity(ε_3)

        C3 = positive_damping(θ)

        C4 = asymptotic_motion(v_ed) + asymptotic_motion(v_uv)

        C5 = asym_dissip(x_ed, v_ed, k, m) + asym_dissip(x_uv, v_uv, k, m)

        #C6 = free_vib_displ(θ, x_ed, (x0, 0), k, m, t_fv) +
        #     free_vib_displ(θ, x_uv, (0, v0), k, m, t_fv)
        C6 = 0

        C7 = initial_elasticity(x_1, uy) + initial_elasticity(x_2, uy) +
             initial_elasticity(x_3, uy)

        C8 = Drucker(fr, ε)     + Drucker(fr_ed, ε_ed) + Drucker(fr_uv, ε_uv) +
             Drucker(fr_1, ε_1) + Drucker(fr_2, ε_2)   + Drucker(fr_3, ε_3)

        C9 = Iliushyn(x, ε)     + Iliushyn(x_ed, ε_ed) + Iliushyn(x_uv, ε_uv) +
             Iliushyn(x_1, ε_1) + Iliushyn(x_2, ε_2)   + Iliushyn(x_3, ε_3)


        # compute the loss
        loss = loss_x + loss_ε + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9

    # otherwise, return an infinity loss value:
    else
        loss = Inf
    end


    # return the loss value
    return loss
end