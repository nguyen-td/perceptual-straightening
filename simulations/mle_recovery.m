%% Setup
clearvars;
clear all
close all
clc

addpath(genpath('/Users/nguyentiendung/GitHub/perceptual-straightening/'))
data_path = '/Users/nguyentiendung/GitHub/perceptual-straightening/data';

%% Run recovery analysis
sim_idx = 1; % which out of the 100 simulated trajectories to use
n_traj = 200;
c_true = zeros(1, n_traj);
c_est = zeros(1, n_traj);

for itraj = 1:n_traj

    % load data
    load(['sim_' sprintf('%04d', itraj) '.mat']);
    c_true(itraj) = rad2deg(mean(S{sim_idx}.c));
    
    nObsCor = S{sim_idx}.Pc_reshaped .* S{sim_idx}.num_trials_mat; 
    nObsTotal = S{sim_idx}.num_trials_mat;
    n_frames = size(S{sim_idx}.Pc_reshaped, 1);
    n_dim = 10;

    [~, ~, c] = ml_fit(S);
    c_est(itraj) = mean(rad2deg(c));

    if mod(itraj, 10) == 1
        disp(['Trajectory: ' num2str(itraj)])
    end
end

%% Visualization
plot(c_est, c_true, 'go')
hold on;
plot([0 180], [0 180], 'k--')
xlabel('Estimated curvature')
ylabel('Ground truth curvature')



