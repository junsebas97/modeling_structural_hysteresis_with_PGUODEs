box(a, b, t) = (sign(t - a) - sign(t - b))/2  # unit-box function

"""
Generate a loading function with the form of test method B of the ASTM 2126

p(t) = astm_load(p_max, T)

ARGS:
p_max: maximum force [kN]    
T:     cycle period   [s]

RETURNS:
p: loading function
"""
function astm_load(p_max, T)
    # amplitude of the load (see Table 2 ASTM 2126-11) - modulation function
    A(t) = 0.0125*box( 0*T,  1*T, t) +    # one   cycle  with 1.25%,
            0.025*box( 1*T,  2*T, t) +    # one   cycle  with  2.5%,
             0.05*box( 2*T,  3*T, t) +    # one   cycle  with    5%,
            0.075*box( 3*T,  4*T, t) +    # one   cycle  with  7.5%,
             0.10*box( 4*T,  5*T, t) +    # one   cycle  with   10%,
             0.20*box( 5*T,  8*T, t) +    # three cycles with   20%,
             0.40*box( 8*T, 11*T, t) +    # three cycles with   40%,
             0.60*box(11*T, 14*T, t) +    # three cycles with   60%,
             0.80*box(14*T, 17*T, t) +    # three cycles with   80%, and
             1.00*box(17*T, 20*T, t)      # three cycles with  100%.

    p(t) = p_max*A(t)*sin(t*2*π/T)

    return p
end