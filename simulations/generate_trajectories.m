% Wrapper script to generate and save trajectories.

%% Setup
clear;
close all;
clc;

% addpath('/Users/nguyentiendung/GitHub/perceptual-straightening/simulations')
addpath('/Users/tn22693/GitHub/perceptual-straightening/simulations')
% save_data_path = '/Users/nguyentiendung/GitHub/perceptual-straightening/data';
% save_data_path = '/Users/tn22693/GitHub/perceptual-straightening/data';
save_data_path = '/Users/gorislab/Documents/GitHub/perceptual-straightening/data/sim';

%% Set trajectory generation settings

% number of simulations for a fixed set of parameter combinations
n_sims_per_param = 100; 

% determines the first file number, e.g., if 15, the first script is saved as 'sim_015.mat'
file_number = 1;

% files from 1082 are simulated with 100 trials

%% Set parameters 

n_frames = 11;
n_reps = 10;
abort_prob = 10;

% d_mu = 1;
d_mu = 1:1:3;
d_sigma = 0.75;

% c_mu = 45;
c_mu = 0:20:180;
c_sigma = 0:10:50;

a_mu = 0;
a_sigma = 1:2:11; % a_sigma > 0

%% Simulate trajectories and save trajectories
for iparam_a_sigma = 1:numel(a_sigma)
    for iparam_d_mu = 1:numel(d_mu)
        for iparam_c_sigma = 1:numel(c_sigma)
            for iparam_c_mu = 1:numel(c_mu)
                clear S
                n_dim = randi([2, n_frames - 1]); % randomly generate dimensions

                for isim = 1:n_sims_per_param
                    [Pc_reshaped, num_trials_mat, x, d, c_true, c_est, a, v_hat] = simulation(n_frames, n_reps, n_dim, abort_prob, d_mu(iparam_d_mu), d_sigma, c_mu(iparam_c_mu), c_sigma(iparam_c_sigma), a_mu, a_sigma(iparam_a_sigma));
                    S{isim}.Pc_reshaped     = Pc_reshaped;
                    S{isim}.num_trials_mat  = num_trials_mat;
                    S{isim}.n_reps          = n_reps;
                    S{isim}.abort_prob      = abort_prob;
            
                    S{isim}.x      = x;
                    S{isim}.d      = d;
                    S{isim}.c      = c_est;
                    S{isim}.c_true = c_true;
                    S{isim}.a      = a;
                    S{isim}.v_hat  = v_hat;
                    
                    % make sure to use 'iparam' to save
                    S{isim}.generative_params.n_frames   = n_frames;
                    S{isim}.generative_params.n_reps     = n_reps;
                    S{isim}.generative_params.n_dim      = n_dim;
                    S{isim}.generative_params.abort_prob = abort_prob;
                    S{isim}.generative_params.d_mu       = d_mu(iparam_d_mu);
                    S{isim}.generative_params.d_sigma    = d_sigma;
                    S{isim}.generative_params.c_mu       = c_mu(iparam_c_mu); 
                    S{isim}.generative_params.c_sigma    = c_sigma(iparam_c_sigma);
                    S{isim}.generative_params.a_mu       = a_mu;
                    S{isim}.generative_params.a_sigma    = a_sigma(iparam_a_sigma);
            
                end
                save(fullfile(save_data_path, ['sim_' sprintf('%04d', file_number) '.mat']), 'S')
                % disp('Saved')
                file_number = file_number + 1;
            end
        end
    end
end