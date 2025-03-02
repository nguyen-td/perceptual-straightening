import torch

def orthogonalize_acc(a, v_hat):
    """
    Function to orthogonalize the acceleration w.r.t. the previous displacement vector, s.t. a(t) ⊥ v(t-1).

    Inputs:
    -------
    a: (n_samples x n_dim) Torch tensor
        Acceleration vector
    v_hat: (n_samples x n_dim) Torch tensor
        Displacement vector

    Output:
    -------
    a_hat_orth: (n_samples x n_dim) Torch tensor
        Orthogonal displacement vector
    """

    # proj_a2v = (a.to(v_hat.dtype) @ v_hat) / (v_hat @ v_hat) * v_hat
    proj_a2v = (torch.sum(a.to(v_hat.dtype) * v_hat, dim=-1) / torch.sum(v_hat * v_hat, dim=-1))[:, None] * v_hat
    a_hat_orth = a - proj_a2v
    # a_hat_orth = a_hat_orth / torch.linalg.norm(a_hat_orth)
    a_hat_orth = a_hat_orth / torch.linalg.norm(a_hat_orth, dim=1)[:, None]

    # assert abs(v_hat @ a_hat_orth) <= 1e-4, "Orthogonality check failed"

    return a_hat_orth

def compute_trajectory(n_samples, n_frames, n_dim, d, c, a_init):
    """
    Compute perceptual trajectory from polar coordinates.

    Inputs:
    -------
    n_samples: Scalar
        Number of samples 
    n_frames: Scalar
        Number of frames
    n_dim: Scalar
        Number of dimensions
    d: (n_samples x n_frames - 1) Torch tensor
        Distance vector
    c: (n_samples x n_frames - 2) Torch tensor
        Curvature vector
    a_init: (n_samples x n_dim x n_frames) Torch tensor
        Initialized acceleration vectors

    Outputs:
    --------
    x: (n_samples x n_dim x n_frames) Torch tensor
        Perceptual locations
    v: (n_samples x n_dim x n_frames - 1) Torch tensor
        Displacement vectors (not normalized)
    c_est: (n_samples x n_frames - 2) Torch tensor
        Estimated curvature vector from generated trajectory; can be used to check if the estimated curvature is the same as the ground truth curvature used to generate the trajectory
    a_hat_orth: (n_samples x n_dim x n_frames - 2) Torch tensor
        Orthonormal acceleration vectors
    v_hat: (n_samples x n_dim x n_frames - 1) Torch tensor
        Normalized displacement vectors
    """

    v_hat = torch.zeros(n_samples, n_dim, n_frames)
    v_hat[:, 0, 0] = 1

    c_est = torch.zeros(n_samples, n_frames - 2)
    a_hat_orth = torch.zeros_like(a_init)

    # step 1: get normalized displacement vector
    for t in range(1, n_frames - 1):
        a_hat_orth[:, :, t - 1] = orthogonalize_acc(a_init[:, :, t - 1], v_hat[:, :, t - 1].clone())

        v_hat[:, :, t] = torch.cos(c[:, t - 1])[:, None] * v_hat[:, :, t - 1].clone() + torch.sin(c[:, t - 1])[:, None] * a_hat_orth[:, :, t - 1].clone()
        v_hat[:, :, t] = v_hat[:, :, t].clone() / torch.linalg.norm(v_hat[:, :, t].clone(), dim=1)[:, None]

        # c_est[:, t - 1] = torch.acos(v_hat[:, t - 1] @ v_hat[:, t])
        c_est[:, t - 1] = torch.acos(torch.sum(v_hat[:, :, t - 1] * v_hat[:, :, t], dim=-1))
    
    # step 2: get displacement vector and perceptual locations
    v = torch.zeros((n_samples, n_dim, n_frames - 1))
    x = torch.zeros((n_samples, n_dim, n_frames))
    for t in range(n_frames - 1):
        v[:, :, t] = d[:, t][:, None] * v_hat[:, :, t]
        x[:, :, t + 1] = x[:, :, t] + v[:, :, t]

    return x, v, c_est, v_hat, a_hat_orth