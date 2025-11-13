# Perceptual curvature estimation 

This repository contains code for the perceptual curvature estimation algorithm introduced and used in the following papers 📚:

> [1] Hénaff, O. J., Goris, R. L., & Simoncelli, E. P. (2019). Perceptual straightening of natural videos. Nature neuroscience, 22(6), 984-991.
> 
> [2] Hénaff, O. J., Bai, Y., Charlton, J. A., Nauhaus, I., Simoncelli, E. P., & Goris, R. L. (2021). Primary visual cortex straightens natural video trajectories. Nature communications, 12(1), 5982.
>
> [3] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), 859-877.

The last reference is a well-written and useful review of the framework that is being used.

💡The original Lua implementation can be found here: [Olivier Hénaff's repository](https://github.com/olivierhenaff/perceptual-straightening/tree/master). 

## Quick start

- [main.ipynb](https://github.com/nguyen-td/perceptual-straightening/blob/main/main.ipynb): Estimate curvature of a single simulated trajectory and visualize the training progress.
- [compute_perc_curvature.py](https://github.com/nguyen-td/perceptual-straightening/blob/main/tests/compute_perc_curvature.py): Same as above but more condensed as a Python script (.py file) without visualization.

Here is a skeleton for a generic curvature estimation task:

```python
import modules.elbo import ELBO

# Load data
n_corr_obs = ...  # (n_frames x n_frames) Matrix (NumPy array) where each entry corresponds to the number of correct observations/trials in the AXB task
n_total_obs = ... # (n_frames x n_frames) Matrix (NumPy array) where each entry corresponds to the number of total observations/trials in the AXB task

# Run inference
n_dim = ...        # Dimensionality of the perceptual (d') space where the trajectory lives in
n_starts = ...     # Number of starts for the multistart procedure, used for initialization
n_iterations = ... # Number of maximal inference iterations for parameter convergence

elbo = ELBO(n_dim, n_corr_obs, n_total_obs, n_starts=n_starts, n_iterations=n_iterations)
_, _, _, _, _, _, _, _, _, _, _, c_est = elbo.optimize_ELBO_SGD() # c_est contains the estimated curvatures in degrees

```

# Repository structure
After cloning or forking this repository, your project layout should look as follows:
```
perceptual-straightening
├── data_analysis/ 							
├── modules/ 											
├── simulations/
├── tests/	
├── utils/						
└── /.../ 					
```

`modules/` contains the main chunk of the low-level code. `elbo.py` contains the low-level code of the hierarchical algorithm.

`simulations` contains MATLAB code to simulate perceptual trajectories. `simulation.m` is a standalone function to simulate one trajectory - used in `generate_trajectories.m` to create a dataset of trajectories.

`utils` contains some other small useful functions called from other functions.

# Variational inference
Direct curvature estimation amounts to maximizing the likelihood of the parameters, $\theta = \left(d^\*, c^\*, \sigma_d, \sigma_c, \Sigma_a \right)$ governing the random variables $z = \left(z_t^d, z_t^c, z_t^a, z^{\lambda} \right),$ that best account for the data. That is,

$$
\theta^* = argmax_{\theta} \\ p(z \mid n, m) = argmax_{\theta} \\ \frac{p(n,m \mid z) p_{\theta}(z)}{p(n,m)}
$$

In the variational Bayesian inference framework, the goal is to numerically approximate the intractible evidence 

$$
log p_{\theta}(n,m) = log \\ \int p(n,m \mid z) \ p_{\theta}(z) \ dz
$$ 

by introducing a variational posterior $q_{\phi}(z | n,m)$ (cf. [1,3]), where $(m,n)$ is the data containing the number of correct and incorrect responses in the AXB task. This probability is approximated by the evidence lower bound (ELBO): 

$$
log p_{\theta}(n,m) \geq \mathbb{E}_{q\_{\phi}(z|n,m)}[log p(n,m | z)] - D\_{KL} \left( q\_{\phi}(z | n,m) \ \rVert \ p\_{\theta}(z) \right)
$$ 

$$
log p_{\theta}(n,m) \geq ELBO(q)
$$ 

## Algorithm
0. Initialize the prior $p\_{\theta}(z)$ and the variational posterior $q\_{\phi}(z | n,m)$.
1. Compute the KL-divergence term.
2. Sample $n$ samples from the variational posterior $z_i \sim q\_{\phi}(z | n,m), \quad i = 1, ..., n$.
3. Use $z$, which contains information about $(d, c, a, \lambda)$, to construct the trajectory and compute the expected likelihood $\frac{1}{n} \sum_i^n log p(n,m | z_i)$. Note that, in the AXB task, the likelihood is governed by a binomial distribution $B(n_{ij},m_{ij} | p_{ij})$ describing the subject's number of correct and incorrect responses.
5. Compute the ELBO term.
6. Compute the gradient and update the parameters using the Adam optimizer.
7. Return to Step 1 until convergence.

## Choice of the prior and posterior
To avoid ambiguity, vectors and matrices are now denoted using **bold** letters.

Both the prior and posterior are chosen to follow a Gaussian distribution. As such, we have a prior $p_{\boldsymbol{\theta}}(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, where $\boldsymbol{\mu} = \left(\mu_{d^\*}, \mu_{c^\*}, \mu_{\boldsymbol{a}^\*}, \mu_{\lambda^\*}\right)^T$ (note that $\boldsymbol{a}$ is a vector) and 

$$\boldsymbol{\Sigma} = \begin{bmatrix}
\sigma_{d^\*} & 0   & 0   & 0 \\
0   & \sigma_{c^\*} & 0   & 0 \\
0   & 0   & \sigma_{\boldsymbol{a^\*}} & 0 \\
0   & 0   & 0   & \sigma_{\lambda^\*}
\end{bmatrix}.$$

Please refer to the current code for which values are currently used for the initialization. Here, $\boldsymbol{\theta}$ referes to all *learnable* parameters, which can be specified. For example, in [1], only $\boldsymbol{\theta} = \left(\mu_{d^\*}, \mu_{c^*\}, \sigma_{d^\*}, \sigma_{c^\*}, \boldsymbol{\Sigma_{a}^\*}\right)$ were chosen to be learnable. A learnable parameter can be specified by setting `requires_grad=True`. The prior shapes the *global* parameters, i.e., the global distance, the global curvature etc.

The posterior shapes the *local* parameters, i.e., we have one posterior distribution for each node $T$ with variables that are defined over the following spaces:

$$\lambda \in \mathbb{R}$$
$$d_t \in \mathbb{R}, \quad t = 1, \ldots, T$$
$$c_t \in \mathbb{R}, \quad t = 2, \ldots, T$$
$$\boldsymbol{a_t} \in \mathbb{R}^{(N-1)}, \quad t = 2, \ldots, T$$
$$\boldsymbol{v_t} \in \mathbb{R}^{(N-1)}, \quad t = 1, \ldots, T$$
$$\boldsymbol{x_t} \in \mathbb{R}^{(N-1)}, \quad t = 0, \ldots, T,$$

where $N$ refers to the number of dimensions, i.e., $T = N - 1$.

Each of the $T$ individual posteriors are governed by their own set of means and covariances similar to the prior, i.e., $\boldsymbol{\mu}\_t = \left(\mu_{d_t}, \mu_{c_t}, \mu_{\boldsymbol{a}\_t}, \mu_{\lambda}\right)^T$, and analogously for the covariance matrix. Similarly, $\boldsymbol{\phi}$ refers to the posterior parameters that are *learnable*.





