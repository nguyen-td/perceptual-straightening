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
%   Pc_reshaped  - [n_frames x n_frames] performance matrix, contains proportion of correct responses for each pair
%   reps_mat     - [n_frames x n_frames] matrix of repetitions per condition, containing the number of trials per condition
%   x            - [n_dim x n_frames] perceptual locations/trajectory
%   d            - [n_frames - 1] vector of local distances
%   c            - [n_frames - 2] vector of local curvatures
%   a            - [n_dim x n_frames] matrix of local accelerations

function [Pc_reshaped, resp_mat, x, d, c, a] = simulation(n_frames, n_reps, n_dim, abort_prob, d_mu, d_sigma, c_mu, c_sigma, a_mu, a_sigma)
    
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
    
    c = zeros(1, ExpParam.numFrames - 2);
    
    % Step 1: Get normalized displacement vector
    for t = 2:ExpParam.numFrames - 1
    
        proj_a2v = ((ExpParam.a(:, t-1)' * v_hat(:, t-1)) / (v_hat(:, t-1)' * v_hat(:, t-1))) * v_hat(:, t-1);
        a_hat_orth = ExpParam.a(:, t-1) - proj_a2v;
        a_hat_orth = a_hat_orth / norm(a_hat_orth);
        assert(abs(v_hat(:, t-1)' * a_hat_orth) <= 1e-6)
        
        v_hat(:, t) = cos(ExpParam.c(t-1)) * v_hat(:, t-1) + sin(ExpParam.c(t-1)) * a_hat_orth; 
        v_hat(:, t) = v_hat(:, t) / norm(v_hat(:, t));    
        
        c(t-1) = acos(v_hat(:, t-1)' * v_hat(:, t));
    end
    
    % Step 2: Get displacement vector and perceptual coordinates
    v = zeros(ExpParam.numDim, ExpParam.numFrames - 1);
    for t = 1:ExpParam.numFrames - 1
        v(:, t) = ExpParam.d(t) * v_hat(:, t);
    end
    
    % Step 3: Get perceptual locations
    x = zeros(ExpParam.numDim, ExpParam.numFrames);
    for t = 1:ExpParam.numFrames - 1
        x(:, t+1) = x(:, t) + v(:, t);
    end
        
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
            
    % calculate proportion correct and number of completed trials
    Pc = NaN(1,length(ExpParam.all_pairs));
    resp_mat = zeros(1, length(ExpParam.all_pairs));
    for i = 1:length(ExpParam.all_pairs)
        Pc(i) = sum(trial_mat(:,i))/ExpParam.numReps; % proportion correct
        resp_mat(i) = sum(trial_mat(:, i));           % number of trials
    end
    Pc_reshaped = reshape(Pc,ExpParam.numFrames,ExpParam.numFrames);

    % abort [abort_prob]% of the trials
    n_aborted_trials = floor((abort_prob / 100) * sum(resp_mat));
    rand_cond = randi(numel(resp_mat), n_aborted_trials, 1);
    aborted_trials = accumarray(rand_cond, 1, [numel(resp_mat), 1]);
    % aborted_trials = zeros(numel(resp_mat), 1);
    % count = 0;
    % for iabort = 1:n_aborted_trials
    %     rand_cond = randi(n_aborted_trials);
    %     aborted_trials(rand_cond) = aborted_trials(rand_cond) + 1;
    %     count = count + 1;
    % 
    %     if count >= n_aborted_trials
    %         break
    %     end
    % end
    resp_mat = resp_mat - aborted_trials';
    resp_mat = reshape(resp_mat, ExpParam.numFrames, ExpParam.numFrames);

    % for saving data
    d = ExpParam.d;
    a = ExpParam.a;
    
    % % Visualize the simulated data
    % subplot(1,3,1)
    % plot(x(1, :), x(2, :), 'ko-', 'markersize', 12, 'markerfacecolor', [1 0 0], 'linewidth', 1)
    % hold on, box off, axis equal, axis square
    % 
    % subplot(1,3,2)
    % plot(ExpParam.c, c, 'r+', 'linewidth', 1)
    % hold on, box off, axis square
    % plot([0 pi], [0 pi], 'k--')
    % plot(mean(ExpParam.c), mean(c), 'ko', 'MarkerSize', 12, 'markerfacecolor', [0 .5 1])
    % 
    % subplot(1,3,3)
    % imagesc(Pc_reshaped);
    % hold on, axis square, box off
    % colormap('gray'); 
    % cBar = colorbar; 
    % title(cBar,'Proportion correct'); 
    % cBar.Ticks = [0.35,1];
    % xticks([1 ExpParam.numFrames]); yticks([1 ExpParam.numFrames]);
    % xlim([1 ExpParam.numFrames]); ylim([1 ExpParam.numFrames]);
    % xlabel('Frame number'); ylabel('Frame number');
    % titleText = sprintf('Simulated discriminability, %d trials', ExpParam.numReps);
    % title(titleText);
    % set(gca, 'FontName', 'Arial');
    % set(gca, 'FontSize', 12);
    % 
    % intention = c_mu
    % reality = rad2deg(mean(c))
    % performance = mean(Pc)
end
