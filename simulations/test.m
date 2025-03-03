%% Test compute trajectory
clear 
clc

n_frames = 11;
n_dim = 10;

nObsCor = ones(n_frames, n_frames) * 6;
nObsTotal = ones(n_frames, n_frames) * 10;

d = ones(n_frames - 1, 1);
c = ones(n_frames - 2, 1) * 70;
a_init = ones(n_dim, n_frames) * (-2);
v_hat = zeros(n_dim, n_frames);
v_hat(1,1) = 1;

[x, v, c_est, a_hat_orth, v_hat] = compute_trajectory(d, c, a_init, v_hat, n_frames, n_dim);

% remove for-loop
% Get perceptual distances
distance    = @(p1,p2) sqrt(sum((p1-p2).^2));
for ir = 1:n_frames
    for ic = 1:n_frames
        d(ir, ic) = distance(x(:,ir), x(:, ic));
    end
end

% Get predicted probability correct
probCor = normcdf(d./sqrt(2)) .* normcdf(d ./ 2) + normcdf(-d./sqrt(2)) .* normcdf(-d ./ 2);
    
% Get LLH
NLL = -sum(sum((nObsCor .* log(probCor)) + (nObsTotal - nObsCor) .* log(1 - probCor)));  