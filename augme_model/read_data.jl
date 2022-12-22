using DelimitedFiles, Interpolations

"""
In three consecutive points (x1, x2, x3), this function assesses the
perpendicular distance between the mid point x2 and the line that joints the
outer points (x1, x2)

perpendicular_dist(x1, x2, x3)

ARGS:
x1, x3: coordinates of outer points
x2:     coordinates of the mid point

RETURNS:
h: perpendicular distance between the mid point x2 and the line that joints the
   outer points
"""
function perpendicular_dist(x1, x2, x3)
    # compute the distances between the points
    d12 = sqrt((x1 - x2)'*(x1 -x2))
    d13 = sqrt((x1 - x3)'*(x1 -x3))
    d23 = sqrt((x2 - x3)'*(x2 -x3))

    # compute the semiperimeter, area, and height of the triangle formed by the
    # three points
    s = (d12 + d13 + d23)/2
    A = sqrt(abs(s*(s - d12)*(s - d13)*(s - d23)))
    h = 2*A/d13

    # since the line between the outer points was assumed to be the triangle's
    # base, the height is the perpendicular distance of the mid point
    return h
end



"""
This function downsamples a signal y in such a way that the distance between the
linear interpolation of the downsampled signal and the original signal does not
exceed a treshold

idx = max_deviation_downsampler(y, treshold=0.1)

ARGS:
y:         original signal
threshold: distance threshold

RETURNS:
idx: indexes of the records used for the downsampled signal
"""
function max_deviation_downsampler(y, threshold=0.1)
    n_dat  = size(y, 1)

    # do not domsample when the threshold is zero
    if threshold == 0
        return 1:n_dat
    end

    # otherwise:
    # the downsampled series has the same initial and end points of the original
    # series
    idx = [1; n_dat]

    last_i = 1    # latest point added to the downsampled series


    for i = 2:n_dat
        # take the mid point between the current point and the last point added
        # to the downsampled series
        idx_mp = round(Int, (i + last_i)/2)

        if perpendicular_dist(y[last_i, :], y[idx_mp, :], y[i, :]) > threshold
            push!(idx, i - 1)
            last_i = i - 1
        end
    end
    
    return idx
end




"""
Estimate the envelope of a time series by linearly interpolating between
magnitude peaks

Si = envelope(t, y)

ARGS:
t: evaluation times    {t = ti}
y: data                {y(ti)}

RETURNS:
Si: envelope's value at each t
"""
function envelope(t, y)
    n_pt  = length(y)
    y_abs = abs.(y)    # consider only the magnitudes

    # the peaks and the limits of the series are extracted:
    env_t = [t[1]];    env_y = [y_abs[1]]                     # **start**

    for i = 2:(n_pt - 1)                                      # **peaks**
        if y_abs[i - 1] < y_abs[i] >= y_abs[i + 1]
            push!(env_t, t[i])
            push!(env_y, y_abs[i])
        end
    end

    # at the end of the series either:        
    if y_abs[n_pt - 1] ≤ y_abs[n_pt]                          # **end**
        # a) interpolate the envelope to the final data
        push!(env_t, t[n_pt]);
        push!(env_y, y_abs[n_pt])
    else # or
        # b) keep the envelope constant
        push!(env_t, t[n_pt])
        push!(env_y, last(env_y))
    end

    # the peaks are interpolated to compute the envelope and then, it is
    # evaluated at each time step
    S = LinearInterpolation(env_t, env_y)
    return S.(t)
end



""""
read the experimental data and set the corresponding variables

example, θ_sys = read_data(dir, frac, samp_rate)

ARGS:
dir:       directory where data is stored
frac:      portion of data to extract and process (50%, 80%, etc).
           Default = 1 (100%)
max_dev: downsampling parameters. Maximum deviation between the downsampled 
         the original signals. Default = 0.0

RETURNS:
example: example data, that is, the input, target, and envelopes, where:
            - the input has the time "t", the external force "p(t)", and the 
              initial conditions "u0",
            - the target corresponds to recorded displacements "x(t)" and
              computed dissipated energy "ε(t)", and
            - the envelopes are the displacement "Sx(t)" and dissipated energy
              "Sε(t)" envelopes.
θ_sys:   system parameters, that is, the mass "m", initial stiffness "k", and
         yielding displacement "uy".
"""
function read_data(dir, frac=1.0, max_dev=0.0)
    # read the experimental data
    p        = readdlm("$(dir)/load_kN.txt")[:]            # [kN]
    x        = readdlm("$(dir)/displ_mm.txt")[:]           # [mm]
    t        = readdlm("$(dir)/time_s.txt")[:]             # [s]
    u0       = readdlm("$(dir)/initial_condition.txt")[:]  # u0 = [x0 v0 fr0 ε0]
    m, k, uy = readdlm("$(dir)/properties.txt")            # [kg], [kN/mm], [mm]

    # extract the fraction of interest
    n = round(Int, length(p)*frac)    # number of readings
    p = p[1:n]
    x = x[1:n]
    t = t[1:n]

    # downsample data
    idx_x = max_deviation_downsampler([t  x], max_dev)
    idx_p = max_deviation_downsampler([t  p], max_dev)
    idx   = sort(union(idx_x, idx_p))
    p     = p[idx]
    x     = x[idx]
    t     = t[idx]

    n = length(t)    # number of readings that are used

    # calculate the dissipated energy 
    Δε = [(p[i] + p[i - 1])*(x[i] - x[i - 1])/2 for i = 2:n]   # [kN]*[mm] = [J]
    ε  = cumsum([0; Δε])./m                                    # [J/kg]

    # convert the records of p(t) to a function and compute the envelopes of the
    # displacement and dissipated energy
    pt = LinearInterpolation(t, p)
    Sx = envelope(t, x)
    Sε = envelope(t, ε)

    # create the example (input, target, and envelopes) and return it together
    # with the system parameters
    input   = Dict("t"     =>     t,   "p(t)" =>    pt,        "u0"  =>    u0)
    target  = Dict("x(t)"  =>     x,   "ε(t)" =>     ε)
    envel   = Dict("Sx(t)" =>    Sx,  "Sε(t)" =>    Sε)
    example = Dict("input" => input, "target" => target, "envelopes" => envel)

    θ_sys   = Dict("m" => m, "k" => k, "uy" => uy)

    return example, θ_sys
end