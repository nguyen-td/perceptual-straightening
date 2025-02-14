% This script simulates the behavioral responses of the AxB task in the 
% temporal straightening paper (2019)
%
% Inputs:
%   n_frames     - number of frames
%   n_reps       - number of repetitions per condition, e.g., 10 repetitions for each frame pair combination
%   abort_prob   - in percent, probability of aborted trials, e.g., 10%
%   n_dim        - number of simulated dimensions
%   d_mu         - mean of the Gaussian distribution around d
%   d_sigma      - variance of the Gaussian distribution around d
%   c_mu         - mean of the Gaussian distribution around c
%   c_sigma      - variance of the Gaussian distribution around c
%   a_mu         - mean of the Gaussian distribution around a
%   a_sigma      - variance of the Gaussian distribution around a
%
% Outputs:
%   Pc_reshaped    - [n_frames x n_frames]    performance matrix, contains proportion of correct responses for each pair
%   num_trials_mat - [n_frames x n_frames]    matrix of repetitions per condition, containing the number of trials per condition
%   x              - [n_dim x n_frames]       perceptual locations/trajectory
%   d              - [n_frames - 1]           vector of local distances
%   c              - [n_frames - 2]           vector of local curvatures
%   a              - [n_dim x n_frames]       matrix of local accelerations
%   v_hat          - [n_dim x (n_frames - 1)] normalized displacement vectors

function [Pc_reshaped, num_trials_mat, x, d, c_true, c_est, a, v_hat] = simulation(n_frames, n_reps, n_dim, abort_prob, d_mu, d_sigma, c_mu, c_sigma, a_mu, a_sigma)
    
    if n_dim > n_frames
        error('Number of dimensions cannot exceed the number of frames.')
    end

    % Global constants
    ExpParam.numFrames  = n_frames;
    ExpParam.numReps    = n_reps; 
    ExpParam.numDim     = n_dim; % dimensionality of the perceptual space
    
    % d_mu = 1; 
    % d_sigma = 0.75;
    % 
    % c_mu = 20;
    % c_sigma = 15;
    % 
    % a_mu = -3;
    % a_sigma = 2; % needs to be larger than zero
    
    % Generate a perceptual trajectory of N frames
    v_0              = ones(1,ExpParam.numDim)'; % first vector at t0
    ExpParam.vectors = zeros(ExpParam.numDim, ExpParam.numFrames); 
    unit             = @(vec) vec/sqrt(sum(vec.^2));
    
    % sample vector length, curvature, and acceleration
    ExpParam.d = abs(normrnd(d_mu, d_sigma, [ExpParam.numFrames - 1, 1]));
    ExpParam.c = abs(deg2rad(normrnd(c_mu,c_sigma,[ExpParam.numFrames - 2,1])));
    ExpParam.a = normrnd(a_mu, a_sigma, [ExpParam.numDim, ExpParam.numFrames - 2]);
    
    % generate perceptual trajectory
    v_hat = zeros(ExpParam.numDim, ExpParam.numFrames - 1);
    v_hat(1,1) = 1;
    [x, v, c_est, a_orth, v_hat] = compute_trajectory(ExpParam.d, ExpParam.c, ExpParam.a, v_hat, ExpParam.numFrames, ExpParam.numDim);
        
    % Simulate the AXB responses
    % get all pairwise combinations (A,B), X will be identical to A or B
    ExpParam.all_pairs = [repelem(1:ExpParam.numFrames,ExpParam.numFrames)',...
        repmat(1:ExpParam.numFrames,1,ExpParam.numFrames)'];
    
    % simulate responses 
    mus         = x;
    sigma       = 1;
    sigma_mat   = eye(ExpParam.numDim) * sigma;
    distance    = @(p1,p2) sqrt(sum((p1-p2).^2));
    trial_mat    = NaN(ExpParam.numReps,length(ExpParam.all_pairs)); 
    
    for i = 1:length(ExpParam.all_pairs)
        for j = 1:ExpParam.numReps
            if (rand(1) > abort_prob / 100) % trials can randomly be aborted
                simA = mvnrnd(mus(:,ExpParam.all_pairs(i,1)),sigma_mat,1); % frame A
                simB = mvnrnd(mus(:,ExpParam.all_pairs(i,2)),sigma_mat,1); % frame B
                if rem(j,2) == 0
                    simX = mvnrnd(mus(:,ExpParam.all_pairs(i,1)),sigma_mat,1); %draw from A
                    dist_AX = distance(simA,simX);
                    dist_BX = distance(simB,simX);
                        if dist_AX < dist_BX 
                            trial_mat(j,i) = 1; %correct response
                        else
                            trial_mat(j,i) = 0; %incorrect response
                        end
                else
                    simX = mvnrnd(mus(:,ExpParam.all_pairs(i,2)),sigma_mat,1); %draw from B
                    dist_AX = distance(simA,simX);
                    dist_BX = distance(simB,simX);
                        if dist_BX < dist_AX  
                            trial_mat(j,i) = 1; %correct response
                        else
                            trial_mat(j,i) = 0; %incorrect response
                        end
                end 
            end
        end
    end

    % calculate proportion correct and number of completed trials
    Pc = NaN(1,length(ExpParam.all_pairs));
    num_trials_mat = zeros(1, length(ExpParam.all_pairs));
    for i = 1:length(ExpParam.all_pairs)
        Pc(i) = nanmean(trial_mat(:,i)); % proportion correct
        num_trials_mat(i) = sum(isfinite(trial_mat(:, i)));     % number of trials
    end
    Pc_reshaped = reshape(Pc,ExpParam.numFrames,ExpParam.numFrames);
    num_trials_mat = reshape(num_trials_mat, ExpParam.numFrames, ExpParam.numFrames);

    % for saving data
    d      = ExpParam.d;
    a      = a_orth;
    c_true = ExpParam.c;

    % intention = c_mu
    % reality = rad2deg(mean(c))
    % performance = mean(Pc)
end
