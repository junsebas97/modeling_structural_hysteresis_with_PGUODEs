# Modeling hysteretic structures with PGUODEs

This repository stores three hysteretic models and experimental data on hysteretic structural systems. The models use the theory of *Physics-Guided Neural Networks* (PGNNs) [1] and *Universal Ordinary Differential Equations* (UODEs) [2]; thus, they correspond to *Physics-Guided Universal Ordinary Differential Equations* (PGUODEs).

UODEs are differential equations that use universal approximators, such as *multilayer perceptrons* (MLP), on the right-hand side of the equation; for example,

$$ \frac{\mathrm{d}u(t)}{\mathrm{d}t} = f(u, t) + \mathrm{MLP}(u, t),$$

where $f$ is a function of the variables $u$ and $t$, and $\mathrm{MLP}(\cdot)$ represents a multilayer perceptron. On the other hand, PGNNs correspond to artificial neural networks that use physical constraints in their loss function $L$ so that they learn the physics governing the task; for instance,

$$ L = \sum_{i}^{N_{e}} \frac{(y_{i} - \hat{y}_{i})^2}{N_{e}} + H(\hat{y}_{i}),$$

where $y_{i}$ and $\hat{y}_{i}$ stand for the targets and the predictions of the network and $H(\cdot)$ represents a physical constraint.

The proposed models (called [short](./short_model/), [full](./full_model/), and [augmented](./augme_model/)) employ the above algorithms to include MLPs in the dynamic equation of structures and physical principles in their training. The MLPs model the dynamics of restoring force $f_{r}(t)$ based on the displacement $x(t)$, the velocity $v(t)$, the dissipated energy $\varepsilon(t)$, and eventually, on the largest displacement $x_{l}(t)$ and a latent quantity $s(t)$. In this way, [the short model](./short_model/) corresponds to the following ODE

```math
\begin{bmatrix}
    \frac{\mathrm{d}x(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}v(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}f_{r}(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}\varepsilon(t)}{\mathrm{d}t}
\end{bmatrix}
=
\begin{bmatrix}
    v(t) \\
    \frac{1}{m} [p(t) - cv(t) - f_{r}(t)] \\
    \operatorname{MLP}(x(t), \mathrm{sign}(v(t)), f_{r}(t), \varepsilon(t)) \cdot v(t)  \\
    \frac{1}{m} f_{r}(t) v(t)
\end{bmatrix},
```

subjected to the four initial conditions $x(0)$, $v(0)$, $f_{r}(0)$, y $\varepsilon(0)$; here, $m$, $c$, and $p(t)$ stand for the mass, the damping constant, and the external force. Similarly, [the full model](./full_model/) describes a hysteretic structural system as
```math
\begin{bmatrix}
    \frac{\mathrm{d}x(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}v(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}f_{r}(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}\varepsilon(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}x_{l}(t)}{\mathrm{d}t}
\end{bmatrix}
= 
\begin{bmatrix}
        v(t) \\
        \frac{1}{m} [p(t) - cv(t) - f_{r}(t)]            \\
        \operatorname{MLP}(x(t), \mathrm{sign}(v(t)), f_{r}(t), \varepsilon(t), x_{l}(t)) \cdot v(t) \\
        \frac{1}{m} f_{r}(t) v(t)  \\
        |v(t)| \hat{\operatorname{H}}(|x(t)| - x_{l}(t))
\end{bmatrix},
```
where $\hat{H}(\cdot)$ is the Heaviside step function, and the initial condition $x_{l}(0)$ is also required. Finally, [the augmented model](./augme_model/) corresponds to the following ODE:
```math
\begin{bmatrix}
    \frac{\mathrm{d}x(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}v(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}f_{r}(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}\varepsilon(t)}{\mathrm{d}t}  \\
    \frac{\mathrm{d}s(t)}{\mathrm{d}t}
\end{bmatrix}
= 
\begin{bmatrix}
    v(t)                                    \\
    \frac{1}{m} [p(t) - cv(t) - f_{r}(t)]   \\
    O_{1}(t) \cdot v(t)                     \\
    \frac{1}{m} f_{r}(t) v(t)               \\
    O_{2}(t)
\end{bmatrix},
```
where $[O_{1}(t), O_{2}(t)] = \operatorname{MLP}(x(t),\ \mathrm{sign}(v(t)),\ f_{r}(t),\ \varepsilon(t),\ s(t))$ are the output of the multilayer perceptron, and the initial condition $s(0)$ is also required. 

For training, the models use a loss function $L(\theta)$ with a data-driven $l_{D}(\theta)$ and physics-guided component $l_{P}(\theta)$; that is, $L(\theta) = l_{D}(\theta) + l_{P}(\theta)$. The data-driven part corresponds to a modulated mean-squared error between the predicted and recorded displacements and dissipated energies:

$$l_{D}(\theta) = \sum_{i}^{N_{e}} \frac{(x(t_{i}) - \hat{x}(t_{i}))^{2}}{S_{x}(t)} + \frac{(\varepsilon(t_{i}) - \hat{\varepsilon}(t_{i}))^{2}}{S_{\varepsilon}(t)},$$

where $x(t)$ and $\hat{x}(t)$ are the recorded and predicted displacements, $\varepsilon(t)$ and $\hat{\varepsilon}(t)$ are the recorded and predicted dissipated energy, and $S_{x}(t)$ and $S_{\varepsilon}(t)$ are envelopes of the recorded displacement and dissipated energy. On the other hand, the physics-guided loss $l_{P}(\theta)$ encompasses the following principles of structures:

- Newton's motion laws,
- BIBO stability,
- passivity,
- hysteresis dependency on velocity sign, and
- elastoplasticity.

Following the theory of PGNNs [1], these principles are formulated as equality or inequality constraints of the predicted responses.

The proposed models were validated with experimental data of [ferrocement](./tests/gilb_data)[3] and [recycled plastic lumber](./tests/rpl_data)[4] (RPL) walls, showing promising results. In these systems, the models have good accuracy, physical consistency, and generalization. Furthermore, they were versatile because they learned very different hysteresis with complicated nonlinear behaviors. For instance, the following figure shows the prediction of the short model for the ferrocement wall; observe that it produces a physically feasible hysteresis and accurately fits the displacements $x(t)$, dissipated energy $\varepsilon(t)$, and the hysteresis.


For further information, the reader is referred to Delgado-Trujillo (2023) [5], which treats the models, their background theory, and their validation in detail.

### Note:
For the development and validation of the models, we modified the source code of the *sciml_train* function. We pass the gradients of the parameters $\nabla \theta$ to the callback function so that they are considered in the stopping criteria. Nonetheless, this repository implements a second method of the callback function that does not receive the gradients to prevent the user modifies the source code.

## Bibliography:
[1] A. Karpatne, W. Watkins, J. Read, V. Kumar, Physics-guided neural networks (PGNN): An application in lake temperature modeling, arXiv preprint arXiv:1710.11431v2 (2017). https://arxiv.org/abs/1710.11431

[2] C. Rackauckas, Y. Ma, J. Martensen, C. Warner, K. Zubov, R. Supekar, D. Skinner, A. Ramadhan, Universal differential equations for scientific machine learning, arXiv preprint arXiv:2001.04385 (2020). https://arxiv.org/abs/2001.04385.

[3] G. A. Ortiz, D. A. Alvarez, D. Bedoya-Ruiz, Identication of Bouc-Wen type models using multi-objective optimization algorithms, Computers & Structures 114-115 (2013) 121-132.

[4] D. Bedoya-Ruiz, J. E. Hurtado, L. Pujades, Experimental and analytical research on seismic vulnerability of low-cost ferrocement dwelling houses, Structure and Infrastructure Engineering 6 (2010) 55-62.

[5] J. S. Delgado-Trujillo, Modeling of hysteretic structural systems using multilayer perceptrons and physics-guiding techniques, Master's thesis, Universidad Nacional de Colombia, Manizales - Caldas, 2023. **(In Process)**
