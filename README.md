# Perceptual curvature estimation 

This repository contains code for the perceptual curvature estimation algorithm introduced and used in the following papers 📚:

> [1] Hénaff, O. J., Goris, R. L., & Simoncelli, E. P. (2019). Perceptual straightening of natural videos. Nature neuroscience, 22(6), 984-991.
> 
> [2] Hénaff, O. J., Bai, Y., Charlton, J. A., Nauhaus, I., Simoncelli, E. P., & Goris, R. L. (2021). Primary visual cortex straightens natural video trajectories. Nature communications, 12(1), 5982.
>
> [3] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), 859-877.

The last reference is a very well-written and useful review of the framework that is being used.

The code is tested on simulated trajectories where the ground truth curvature is known. The quality of the estimation is tested by performing a recovery analysis (cf. supplementary figure in [1]). That is, the estimated curvature should be the same (or approximately the same) as the true curvature across the whole range of curvatures, i.e., from 0 to $\frac{\pi}{2}$). The figure below shows a biased estimation (a) and an unbiased, desired estimation (b). Panel a) is the result of the greedy, two-step direct estimation. Panel b) shows the results of the most likely perceptual curvature given many plausible perceptual trajectories, i.e., the method to re-implement.

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
In the variational inference framework, the goal is to approximate an intractible posterior $p(x | z)$ with a variational one $q_{\phi}(z)$ (or $q_{\phi}(z | x)$ if it depends on data) that is estimated numerically, where $x$ corresponds to the data and $z$ to the potentially high-dimensional latent (or hidden) variable over which to marginalize (cf. [1,3]). To this end, the evidence lower bound (ELBO) is maximized. Let $(m,n)$ be the data containing the number of correct and incorrect responses in the AXB task, then the goal is to maximize the probability of observing these responses $p_{\theta}(n,m)$, parameterized by a set of parameters $\theta$ (corresponding to the global trajectory parameters). This probability is approximated by a variational posterior distribution, parameterized by its own set of parameters $\phi$ (corresponding to the local estimates of the trajectory parameters): 

$$
log p_{\theta}(n,m) \geq \mathbb{E}_{q\_{\phi}(z|n,m)}[log p(n,m | z)] - D\_{KL} \left( q\_{\phi}(z | n,m) \ \rVert \ p\_{\phi}(z) \right)
$$
