import torch
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm

from modules import compute_trajectory, orthogonalize_acc

# test curvature compuation
n_frames = 11
n_dim = 10
n_samples = 1

n_corr_obs = np.ones((n_frames, n_frames)) * 6
n_total_obs = np.ones((n_frames, n_frames)) * 10

d = torch.ones(1, n_frames - 1)
c = torch.ones(1, n_frames - 2) * 70
a_init = torch.ones(1, n_dim, n_frames) * (-2)
v_hat = torch.zeros(1, n_dim, n_frames)
v_hat[:, 0, 0] = 1
l = 0.06

x, v, c_est, v_hat, a_hat_orth = compute_trajectory(1, n_frames, n_dim, d, c, a_init)

# print(x)
# print(v)
# print(c_est)
# print(a_hat_orth)
# print(v_hat)

 # get perceptual distances
dist = torch.cdist(torch.transpose(x, 1, 2), torch.transpose(x, 1, 2))

# compute hierarchical model
# p_axb = norm.cdf(dist / np.sqrt(2)) * norm.cdf(dist / 2) + norm.cdf(-dist / np.sqrt(2)) * norm.cdf(-dist / 2)
# p = (1 - 2 * l) * p_axb + l
# NLL = -np.sum((n_corr_obs * np.log(p)) + (n_total_obs - n_corr_obs) * np.log(1 - p))

normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # cdf of the standard normal
p_axb = normal.cdf(dist / np.sqrt(2)) * normal.cdf(dist / 2) + normal.cdf(-dist / np.sqrt(2)) * normal.cdf(-dist / 2)
p = (1 - 2 * l) * p_axb.clone() + l
log_ll = torch.sum((torch.tensor(n_corr_obs) * torch.log(p.clone())) + (torch.tensor(n_total_obs) - torch.tensor(n_corr_obs)) * torch.log(1 - p.clone()))
