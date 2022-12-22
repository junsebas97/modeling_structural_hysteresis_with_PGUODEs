"""
compute the system response of the proposed model for a given external 
force p(t) and initial conditions u0

model_response = calc_model_response(p, u0, θ, t, integration_alg,
                                     sensitivity_alg, max_iters)

ARGS:
p:               external force
u0:              initial conditions  [x0, v0, fr0, ε0, xl0]
θ:               model parameters
t:               evaluation times
integration_alg: integration algorithm of the differential equation
sensitivity_alg: method for the calculation of the derivatives
max_iters:       maximum number of iterations of the integration algorithm

The algorithms can be chosen from:
https://diffeq.sciml.ai/stable/solvers/ode_solve/
https://diffeq.sciml.ai/stable/analysis/sensitivity/#Sensitivity-Algorithms

RETURNS: 
model_response: time series of the displacement, velocity, restoring force,
                dissipated energy, and largest displacement
"""
function calc_model_response(p, u0, θ, t, integration_alg=RK4(),
                             sensitivity_alg=ForwardSensitivity(), max_iters=1e6)
    tspan = (t[1], t[end])   # integration time

    # incorporate the external force into the model
    aux_model(du, u, θ, t) = model(du, u, θ, t, p)

    # define and solve the ODE problem
    problem        = ODEProblem(aux_model, u0, tspan, θ)
    model_response = solve(problem, integration_alg, saveat=t,
                           sensealg=sensitivity_alg, maxiters=max_iters)

    return Array(model_response)'
end