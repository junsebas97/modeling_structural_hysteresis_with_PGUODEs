"""
This function returns initial model parameters that complete all the simulations
required when training with physical restrictions

initial_params = good_initialize(example, θ_syst, t_fv)

ARGS:
example: training example contaning the input, the target, and the envelopes
θ_syst:  system parameters
t_fv:    times of the free vibration cases

RETURNS:
initial_params: suitable initial parameters
"""
function good_initialize(example, θ_syst, t_fv)
    while true
        # initialize the model parameters and run the loss
        θ = [                       abs(randn());    # damping
             Random.shuffle(initial_params(MLP))]    # MLP weights

        # For complex models, choose the following line:
        # loss = loss_data(θ, example)

        # For simple models use:
        loss = loss_physics(θ, example, θ_syst, t_fv)

        if loss < Inf    # once the model does not diverge with the random 
            return θ     # initialization, return the model parameters θ
        end
    end
end



"""
this function reports and plots the model prediction for the validation example
and the excitations used in training

print_validation_results(θ_model, example, θ_syst, t_fv)

ARGS:
θ_model: model parameters
example: example of experimental data
θ_syst:  system parameters
t_fv:    times of the free vibration cases 
Fv:      maximum magnitude of the validation load
"""
function print_validation_results(θ_model, example, θ_syst, t_fv, Fv)
    # with the given model parameters, compute the three loss functions and
    # report their values
    l_data     = loss_data(    θ_model, example)
    l_physics  = loss_physics( θ_model, example, θ_syst, t_fv)
    l_extended = loss_extended(θ_model, example, θ_syst, t_fv)

    println("***** SUMMARY: *****

    Data-driven loss: $(round(l_data,     digits = 5))
    Physics loss:     $(round(l_physics,  digits = 5))
    Extended loss:    $(round(l_extended, digits = 5)) \n\n")

    
    
    
    # report the constraint of positive damping
    println(""" *** Loss model parameters *** 
    Possitive damping:  $(round(positive_damping(θ_model), digits=5))""")

    


    # extract data: the targets of the example of experimental data and the
    # system parameters and      
    x_true = example["target"]["x(t)"]
    ε_true = example["target"]["ε(t)"]
    Sx     = example["envelopes"]["Sx(t)"]
    Sε     = example["envelopes"]["Sε(t)"]

    k  = θ_syst["k"]
    uy = θ_syst["uy"]

    # compute the dynamic properties
    ω0 = sqrt(1e6*k/m)
    ξ  = min(1e6*θ_model[1]/(2*m*ω0), 0.999)
    ωD = ω0*sqrt(1 - ξ^2)
        
    

    
    # create the excitations cases:
    # validation example: load pattern 1 of the test A of ASTM 2126:
    val = Dict("u0"   => [0.0, 0.0, 0.0, 0.0, 0.0],
               "t"    => collect(0.0:0.01:60),
               "p(t)" => astm_load(Fv, 3),
               "case" => "validation")

    # experimental data
    exp_dat = Dict("u0"   => example["input"]["u0"],
                   "t"    => example["input"]["t"],
                   "p(t)" => example["input"]["p(t)"],
                   "case" => "experimental data")

    # free vibration from the elastic displacement
    (x0, v0) = (0.95*uy, 0.0)
    fv_ed    = Dict("u0"   => [x0, v0, k*x0, k*(x0^2)/2, x0],
                    "t"    => t_fv,
                    "p(t)" => t -> 0.0*t,
                    "case" => "free vibration from the elastic displacement")

    # free vibration from the unitary velocity
    (x0, v0) = (0.0, 1.0)
    fv_uv    = Dict("u0"   => [x0, v0, k*x0, k*(x0^2)/2, v0],
                    "t"    => t_fv,
                    "p(t)" => t -> 0.0*t,
                    "case" => "free vibration from the unitary velocity")

    # elastic loading up to 25% fy
    el_1 = Dict("u0"   => [0.0, 0.0, 0.0, 0.0, 0.0],
                "t"    => 0.0:0.1:120 ,
                "p(t)" => LinearInterpolation([0, 30, 60, 120], [0, 0.25*k*uy, 0, 0]),
                "case" => "elastic load 25% of yielding")

    # elastic loading up to 50% fy
    el_2 = Dict("u0"   => [0.0, 0.0, 0.0, 0.0, 0.0],
                "t"    => 0.0:0.1:120 ,
                "p(t)" => LinearInterpolation([0., 30, 60, 120], [0, 0.50*k*uy, 0, 0]),
                "case" => "elastic load 50% of yielding")

    # elastic loading up to 50% fy
    el_3 = Dict("u0"   => [0.0, 0.0, 0.0, 0.0, 0.0],
                "t"    => 0.0:0.1:120 ,
                "p(t)" => LinearInterpolation([0., 30, 60, 120], [0, k*uy, 0, 0]),
                "case" => "elastic load 100% of yielding")


    
    # these blank plots are created to draw all the elastic loads responses in
    # one figure
    p1_el = plot();
    p2_el = plot();
    


    # in each excitation:
    for ex in [val, exp_dat, fv_ed, fv_uv, el_1, el_2, el_3]
        # extract the parameters and calculate the model response
        t  = ex["t"]
        u0 = ex["u0"]
        p  = ex["p(t)"]

        model_resp = calc_model_response(p, u0, θ_model, t)
        x  = model_resp[:, 1]
        v  = model_resp[:, 2]
        fr = model_resp[:, 3]
        ε  = model_resp[:, 4]
        xl = model_resp[:, 5]

        
        println("\n***** Losses $(ex["case"]) *****")


        # first, compute and report the constraints of BIBO stability,
        # passivity, and plasticity postulates, which apply to all cases:
        C1 = first_BIBO(x, fr) + second_BIBO(k, xl, fr)
        C2 = passivity(ε)
        C8 = Drucker(fr, ε)
        C9 = Iliushyn(x, ε)

        println("""BIBO stability:     $(round(C1, digits = 5))
                   Passivity:          $(round(C2, digits = 5))
                   Drucker postulate:  $(round(C8, digits = 5))
                   Iliushin postulate: $(round(C9, digits = 5))""")



        # in each excitation case, report the specific constraints and losses
        # and plot the response
        if ex["case"] == "experimental data"
            n_dat = length(t) - 1
            lx    = (1/n_dat)*sum(abs2, (x_true[2:end] .- x[2:end])./Sx[2:end])
            lε    = (1/n_dat)*sum(abs2, (ε_true[2:end] .- ε[2:end])./Sε[2:end])

            # report of the constraints
            println("""loss x(t): $(round(lx, digits = 5))
                       loss ε(t): $(round(lε, digits = 5))""")

            # plot of the response:
            p1 = plot(t,      x, label = "Predicted", linestyle = :dash);
            plot!(p1, t, x_true, label =      "True", xlabel =    "t", ylabel =  "x(t)");

            p2 = plot(t,      ε, label = "Predicted", linestyle = :dash);
            plot!(p2, t, ε_true, label =      "True", xlabel =    "t", ylabel =  "ε(t)");

            p3 = plot(     x, p.(t), label = "Predicted", linestyle = :dash);
            plot!(p3, x_true, p.(t), label =      "True", xlabel = "x(t)", ylabel =  "p(t)");

            p4 = plot(x,    fr, label = "", xlabel = "x(t)", ylabel = "fr(t)");

            display(plot(p1, p2, p3, p4, layout = (2, 2), size = (650, 650)))


        elseif ex["case"] == "free vibration from the elastic displacement" ||
               ex["case"] == "free vibration from the unitary velocity"

            # report of the constraints
            C4 = asymptotic_motion(v)
            C5 = asym_dissip(x, v, k, m)
            C6 = free_vib_displ(θ_model, x, (u0[1], u0[2]), k, m, t_fv)

            println("""Asymptotic motion:           $(round(C4, digits = 5))
                       Asymptotic dissipativity:    $(round(C5, digits = 5))
                       Displacement linear systems: $(round(C6, digits = 5))""")

            # compute the analytical displacements and the stored energy:
            x_an = (u0[1]*cos.(ωD*t_fv) +
                   ((u0[2] + u0[1]*ξ*ω0)/ωD)*sin.(ωD*t_fv)).*exp.(-ξ*ω0*t_fv)
            E    = (1/2)*(1e-6*m)*(v.^2) .+ (1/2)*k*(x.^2)

            # plot of the response:
            p1 = plot(t,    x, label = "Predicted", linestyle = :dash);
            plot!(p1, t, x_an, label = "Analytical", xlabel =    "t", ylabel =   "x(t)");

            p2 = plot(t, log.(E), label = "", xlabel = "t", ylabel =  "log(E(t))");
            display(plot(p1, p2, layout = (1, 2), size = (750, 300)))

        
        elseif ex["case"] == "elastic load 25% of yielding"   ||
               ex["case"] == "elastic load 50% of yielding"   ||
               ex["case"] == "elastic load 100% of yielding"

            # report of the constraints
            C7 = initial_elasticity(x, uy)
            println("Inital elasticity: $(round(C7, digits = 5))")

            # plot of the response
            plot!(p1_el, t,  x, label = "", xlabel =    "t", ylabel =   "x(t)");
            plot!(p2_el, x, fr, label = "", xlabel = "x(t)", ylabel =  "fr(t)");
            display(plot(p1_el, p2_el, layout = (1, 2), size = (750, 300)))

            
        elseif ex["case"] == "validation"
            # plot of the response
            p1 = plot(t,     x, label = "", xlabel =    "t", ylabel =  "x(t)");
            p2 = plot(t,     ε, label = "", xlabel =    "t", ylabel =  "ε(t)");
            p3 = plot(x, p.(t), label = "", xlabel = "x(t)", ylabel =  "p(t)");
            p4 = plot(x,    fr, label = "", xlabel = "x(t)", ylabel = "fr(t)");
            
            display(plot(p1, p2, p3, p4, layout = (2, 2), size = (650, 650)))
        end
    end
end