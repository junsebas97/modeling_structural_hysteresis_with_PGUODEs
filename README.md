# Modeling hysteretic structures with PGUODEs

This repository stores three hysteretic models and experimental data on hysteretic structural systems. The models use the theory of *Physics-Guided Neural Networks* (PGNNs) [1] and *Universal Ordinary Differential Equations* (UODEs) [2]; thus, they correspond to *Physics-Guided Universal Ordinary Differential Equations* (PGUODEs).

UODEs are differential equations that use universal approximators, such as *multilayer perceptrons* (MLP), on the right-hand side of the equation; for example,

$$ \frac{\mathrm{d}u(t)}{\mathrm{d}t} = f(u, t) + \mathrm{MLP}(u, t),$$

where $f$ is a function of the variables $u$ and $t$, and $\mathrm{MLP}(\cdot)$ represents a multilayer perceptron. On the other hand, PGNNs correspond to artificial neural networks that use physical constraints in their loss function $L$ so that they learn the physics governing the task; for instance,

$$ L = \sum_{i}^{N_{e}} \frac{(y_{i} - \hat{y}_{i})^2}{N_{e}} + H(\hat{y}_{i}),$$

where $y_{i}$ and $\hat{y}_{i}$ stand for the targets and the predictions of the network and $H(\cdot)$ represents a physical constraint.

The proposed models (called [short](./short_model/), [full](./full_model/), and [augmented](./augme_model/)) employ the above algorithms to include MLPs in the dynamic equation of structures and physical principles in their training. The MLPs model the dynamics of restoring force $f_{r}(t)$ based on the displacement $x(t)$, the velocity $v(t)$, the dissipated energy $\varepsilon(t)$, and eventually, on the largest displacement $x_{l}(t)$ and a latent quantity $s(t)$. In this way, [the short model](./short_model/) corresponds to the following ODE

$$\begin{bmatrix}
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
\end{bmatrix},$$
subjected to the four initial conditions $x(0)$, $v(0)$, $f_{r}(0)$, y $\varepsilon(0)$; here, $m$, $c$, and $p(t)$ stand for the mass, the damping constant, and the external force. Similarly, [the full model](./full_model/) describes a hysteretic structural system as
$$\begin{bmatrix}
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
\end{bmatrix},$$
where $\hat{H}(\cdot)$ is the Heaviside step function and the initial condition $x_{l}(0)$ is also required. Finally, [the augmented model](./augme_model/) corresponds to the following ODE:
$$\begin{bmatrix}
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
\end{bmatrix},$$
where $[O_{1}(t), O_{2}(t)] = \operatorname{MLP}(x(t),\ \allowbreak \mathrm{sign}(v(t)),\ f_{r}(t),\ \varepsilon(t),\ s(t))$ and the initial condition $s(0)$ is also required. 

For training, the models use a loss function $L(\theta)$ with a data-driven $l_{D}(\theta)$ and physics-guided component $l_{P}(\theta)$; that is, $L(\theta) = l_{D}(\theta) + l_{P}(\theta)$. The data-driven part corresponds to a modulated mean-squared error between the predicted and recorded displacements and dissipated energies:

$$l_{D}(\theta) = \sum_{i}^{N_{e}} \frac{(x(t_{i}) - \hat{x}(t_{i}))^{2}}{S_{x}(t)} + \frac{(\varepsilon(t_{i}) - \hat{\varepsilon}(t_{i}))^{2}}{S_{\varepsilon}(t)},$$

where $x(t)$ and $\hat{x}(t)$ are the recorded and predicted displacements, $\varepsilon(t)$ and $\hat{\varepsilon}(t)$ are the recorded and predicted dissipated energy, and $S_{x}(t)$ and $S_{\varepsilon}(t)$ are envelopes of the recorded displacement and dissipated energy. On the other hand, the physics-guided loss $l_{P}(\theta)$ encompasses the following principles of structures:

- Newton's motion laws,
- BIBO stability,
- passivity,
- hysteresis dependency on velocity sign, and
- elastoplasticity.

Following the theory of PGNNs [1], these principles are formulated as equality or inequality constraints of the predicted responses.

These models employ multilayer perceptrons and laws of nature to identify the hysteretic behavior of structures in an unbiased and physically consistent manner. We validate the proposed framework with experimental data of ferrocement and recycled plastic lumber walls, obtaining good accuracy, generalization and physical consistency. All data and codes used in this work are publicly available on GitHub. The advantage of ANNs is that they are universal approximators that may learn any continuous function from examples; thus, they are a way to accurately identify the hysteresis of a system without the need of simplifications or approximations.

### Note:
For the development and validation of the models, we modified the source code of the *sciml_train* function. We pass the gradients of the parameters $\nabla \theta$ to the callback function so that they are considered in the stopping criteria. Nonetheless, this repository implements a second method of the callback function that does not receive the gradients to prevent the user modifies the source code.

## Bibliography:
[1] A. Karpatne, W. Watkins, J. Read, V. Kumar, Physics-guided neural networks (PGNN): An application in lake temperature modeling, arXiv preprint arXiv:1710.11431v2 (2017). https://arxiv.org/abs/1710.11431

[2] C. Rackauckas, Y. Ma, J. Martensen, C. Warner, K. Zubov, R. Supekar, D. Skinner, A. Ramadhan, Universal differential equations for scientific machine learning, arXiv preprint arXiv:2001.04385 (2020). https://arxiv.org/abs/2001.04385.
