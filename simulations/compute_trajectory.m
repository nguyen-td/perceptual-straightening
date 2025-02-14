% Function to compute the perceptual trajectory (cf. Hénaff, 2019).
%
% Inputs:
%   d      - [n_frames - 1]           distance vector
%   c      - [n_frames - 2]           curvature vector
%   a_init - [n_dim x (n_frames - 2)] initialized acceleration vectors
%   v_hat  - [n_dim x (n_frames - 1)] initialized normalized displacement vectors
%
% Outputs:
%   x          - [n_dim x n_frames]       perceptual locations
%   v          - [n_dim x (n_frames - 1)] displacement vectors (not normalized)
%   c_est      - [n_frames - 2]           estimated curvature vector from generatedtrajectory, can be used to check 
%                                         if the estimated curvature is the same as the ground truth curvature used to generate the trajectory
%   a_hat_orth - [n_dim x (n_frames - 2)] orthonormal acceleration vectors
%   v_hat      - [n_dim x (n_frames - 1)] normalized displacement vectors

function [x, v, c_est, a_hat_orth, v_hat] = compute_trajectory(d, c, a_init, v_hat, n_frames, n_dim)
    
    c_est = zeros(1, n_frames - 2);
    a_hat_orth = zeros(size(a_init));

    % Step 1: Get normalized displacement vector
    for t = 2:n_frames - 1
 
        a_hat_orth(:, t-1) = orthogonalize_acc(a_init(:, t-1), v_hat(:, t-1));
        v_hat(:, t) = cos(c(t-1)) * v_hat(:, t-1) + sin(c(t-1)) * a_hat_orth(:, t-1); 
        v_hat(:, t) = v_hat(:, t) / norm(v_hat(:, t));    
        
        c_est(t-1) = acos(v_hat(:, t-1)' * v_hat(:, t)); % compute curvature of estimated trajectory, can be used for sanity checks
    end
    
    % Step 2: Get displacement vector and perceptual coordinates
    v = zeros(n_dim, n_frames - 1);
    for t = 1:n_frames - 1
        v(:, t) = d(t) * v_hat(:, t);
    end
    
    % Step 3: Get perceptual locations
    x = zeros(n_dim, n_frames);
    for t = 1:n_frames-1
        x(:, t+1) = x(:, t) + v(:, t);
    end

end