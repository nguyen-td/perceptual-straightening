# Perceptual curvature estimation 

This repository contains code for the perceptual curvature estimation algorithm introduced and used in the following papers 📚:

> [1] Hénaff, O. J., Goris, R. L., & Simoncelli, E. P. (2019). Perceptual straightening of natural videos. Nature neuroscience, 22(6), 984-991.
> 
> [2] Hénaff, O. J., Bai, Y., Charlton, J. A., Nauhaus, I., Simoncelli, E. P., & Goris, R. L. (2021). Primary visual cortex straightens natural video trajectories. Nature communications, 12(1), 5982.
>
> [3] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), 859-877.

The last reference is a very well-written and useful review of the framework that is being used.

The code is tested on simulated trajectories where the ground truth curvature is known. The quality of the estimation is tested by performing recovery analyses (cf. supplementary figure in [1]). That is, the estimated curvature should be the same (or approximately the same) as the true curvature across the whole range of curvatures, i.e., from 0 to $\frac{\pi}{2}$). The figure below shows a biased estimation (a) and an unbiased, desired estimation (b). Panel a) is the result of the greedy, two-step direct estimation. Panel b) shows the results of the most likely perceptual curvature given many plausible perceptual trajectories, i.e., the method to re-implement.

<p float="middle">
  <img src="https://github.com/nguyen-td/perceptual-straightening/blob/main/res/recovery_analysis.png?raw=true"/>     
  &nbsp; &nbsp;
</p>

📌 Running the code might require the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), which itself requires the most updated version of MATLAB. We use this to call a MATLAB function directly from Python without the need to translate MATLAB code into Python explicitely. 

## Quick start

- [main.ipynb](https://github.com/nguyen-td/perceptual-straightening/blob/main/main.ipynb): Estimate curvature of a single simulated trajectory and visualize the training progress.
- [direct_estimation.ipynb](https://github.com/nguyen-td/perceptual-straightening/blob/main/direct_estimation.ipynb): Recovery analysis of the biased direct estimation method using maximum likelihood on the observer model (binomial likelihood in [1]).

# Repository structure
After cloning or forking this repository, your project layout should look as follows:
```
perceptual-straightening/	 				
├── data/ 							
├── modules/ 								
├── res/ 					
├── simulations/		
├── utils/						
└── /.../ 					
```

`data/` contains data that can be used for the analysis. Since the current version can generate data during runtime by calling the respective MATLAB function `simulation_py.mat`, this folder may not be necessary. It does, however, contain some of the real data used in previous work.

`modules/` contains the main chunk of the low-level code. `elbo.py` contains the low-level code of the hierarchical algorithm, `recovery_analysis.py` contains code for the recovery analysis, both for the direct estimation of the advanced, unbiased estimation, though it does not work as intended yet. 

`res` contains figures for the repository, i.e., it's not relevant for the actual algorithm but should not be deleted either.

`simulations` contains MATLAB code to simulate perceptual trajectories. `simulation.m` is a standalone function that can also store the trajectories in `/data`. A more flexible way to generate data during runtime is to call `simulation_py.mat` directly from Python.

`utils` contains some other small useful functions called from other functions.

# Current implementation
In the variational inference framework, the goal is to numerically approximate an intractible posterior $p(x | z)$ with a variational one $q_{\phi}(z)$ (or $q_{\phi}(z | x)$ if it depends on data), where $x$ corresponds to the data and $z$ to the potentially high-dimensional latent (or hidden) variable over which to marginalize (cf. [1,3]). To this end, the evidence lower bound (ELBO) is maximized. Let $(m,n)$ be the data containing the number of correct and incorrect responses in the AXB task, then the goal is to maximize the probability of observing these responses $p_{\theta}(n,m)$, parameterized by a set of parameters $\theta$ (corresponding to the global trajectory parameters). This probability is approximated by a variational posterior distribution, parameterized by its own set of parameters $\phi$ (corresponding to the local estimates of the trajectory parameters): 

$$
log p_{\theta}(n,m) \geq \underbrace{\mathbb{E}_{q\_{\phi}(z|n,m)}[log p(n,m | z)] - D\_{KL} \left( q\_{\phi}(z | n,m) \ \rVert \ p\_{\phi}(z) \right)}\_\text{ELBO}
$$

## Algorithm
0. Initialize the prior $p\_{\phi}(z)$ and the variational posterior $q\_{\theta}(z | n,m)$.
1. Compute the KL-divergence term.
2. Sample $n$ samples from the variational posterior $z_i \sim q\_{\theta}(z | n,m), \ i = 1, ..., n$.
3. Use $z$, which contains information about $(d, c, a, \lambda)$, to construct the trajectory and compute the expected likelihood $\frac{1}{n} \sum_i^n log p(n,m | z_i)$. Note that, in the AXB task, the likelihood is governed by a binomial distribution $B(n_{ij},m_{ij} | p_{ij})$ describing the subject's number of correct and incorrect responses.
5. Compute the ELBO term.
6. Compute the gradient and update the parameters using the Adam optimizer.
7. Return to Step 1 until convergence.

## Choice of the prior and posterior
To avoid ambiguity, vectors and matrices are now denoted using **bold** letters.

Both the prior and posterior are chosen to follow a Gaussian distribution. As such, we have a prior $p_{\boldsymbol{\theta}}(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, where $\boldsymbol{\mu} = \left(\mu_{d^\*}, \mu_{d^\*}, \mu_{\boldsymbol{a}^\*}, \mu_{\lambda^\*}\right)^T$ (note that $\boldsymbol{a}$ is a vector) and 

$$\boldsymbol{\Sigma} = \begin{bmatrix}
\sigma_{d^\*} & 0   & 0   & 0 \\
0   & \sigma_{c^\*} & 0   & 0 \\
0   & 0   & \sigma_{\boldsymbol{a^\*}} & 0 \\
0   & 0   & 0   & \sigma_{\lambda^\*}
\end{bmatrix}.$$

Please refer to the current code for which values are currently used for the initialization. Here, $\boldsymbol{\theta}$ referes to all *learnable* parameters, which can be specified. For example, in [1], only $\boldsymbol{\theta} = \left(\mu_{d^\*}, \mu_{c^*\}, \sigma_{d^\*}, \sigma_{c^\*}, \boldsymbol{\Sigma_{a}^\*}\right)$ were chosen to be learnable. A learnable parameter can be specified by setting `requires_grad=True`. The prior shapes the *global* parameters, i.e., the global distance, the global curvature etc.

The posterior shapes the *local* parameters, i.e., we have one posterior distribution for each node - 1. Let T denote the number of nodes, then the variables are defined over the following spaces:

$$\lambda \in \mathbb{R}$$
$$d_t \in \mathbb{R}, \quad t = 1, \ldots, T$$
$$c_t \in \mathbb{R}, \quad t = 2, \ldots, T$$
$$\boldsymbol{a_t} \in \mathbb{R}^{(N-1)}, \quad t = 2, \ldots, T$$
$$\boldsymbol{v_t} \in \mathbb{R}^{(N-1)}, \quad t = 1, \ldots, T$$
$$\boldsymbol{x_t} \in \mathbb{R}^{(N-1)}, \quad t = 0, \ldots, T$$

One assumption that is made in this version is the mean-field approximation [cf. [3]], where the posterior distribution is defined as a *family* of independent distributions. This means that information for each node (distance, curvature, acceleration, lapse rate) comes from an independent posterior. This assumption is not specified in the paper but we believe it is a plausible one (in any case, the code also contains an option where the posterior is defined as a single big multivariate normal). Under this assumption, the posterior is defined as

$$q_{\boldsymbol{\phi}}(\boldsymbol{z | n, m}) = \prod^{(T-1)}_j q_j(\boldsymbol{z}_j | \boldsymbol{n, m}).$$

Each of the $(T-1)$ individual posteriors are governed by their own set of means and covariances similar to the prior, i.e., $\boldsymbol{\mu}\_j^{(t)} = \left(\mu_{d_t}, \mu_{c_t}, \mu_{\boldsymbol{a}\_t}, \mu_{\lambda}\right)^T$, and analogously for the covariance matrix. 

⚠️ **Initialization of the posterior**
> Right now, the means of the posterior distributions are initialized by running the biased, direct two-step maximum likelihood estimation and then taking the biased estimations of the direction, curvature, and acceleration as the initial values for the actual algorithm. This improved the recovery analysis in that the estimation algorithm has become biased itself now. However, it is still an improvement to how it was before (see figures below). We believe that this also explains the dependency of the posterior on data, even if it is technically not necessary, i.e., choosing $q_{\phi}(\boldsymbol{z | n,m})$ over $q_{\phi}(\boldsymbol{z})$ in the paper.

<p float="middle">
  <img src="https://github.com/nguyen-td/perceptual-straightening/blob/main/res/initialization.png?raw=true"/>     
  &nbsp; &nbsp;
</p>

We currently believe that finding out the proper initialization scheme for the prior and posterior will be key to solving this problem. 





