%% Setup
clearvars;
clear all
close all
clc

addpath(genpath('/Users/nguyentiendung/GitHub/perceptual-straightening/'))
load('sim_0001.mat')

% plot_sim_data(S{1}.x, S{1}.c_true, S{1}.c, S{1}.Pc_reshaped, S{1}.n_reps)

%% Step 1
options  = optimset('Display', 'iter', 'MaxIter', 1000, 'MaxFuneval', 10^5);

for irun = 1:1
    objFun   = @(paramVec) giveNLL(paramVec, S{irun});
    
    % define number of frames and number of dimensions
    n_frames = size(S{irun}.Pc_reshaped,1);
    n_dim = size(S{irun}.x,1);
    
    % initialize start vector
    startVec = zeros(n_frames-2 + n_frames-1 + n_dim * (n_frames-2), 1);
    startVec(1:n_frames-2) = abs(deg2rad(normrnd(60, 10, [n_frames-2, 1]))); % c
    startVec(n_frames-1:2*n_frames-3) = abs(normrnd(1, 0.5, [n_frames-1, 1])); % d
    startVec(2*n_frames-2:end) = normrnd(0, 2, [n_dim, n_frames-2]); % a

    % paramVec(1:N-2) = local curvature, starting at frame 1
    % paramVec(N-1:2N-3) = local displacement vector distance
    % paramVec(2N-2:end) = acceleration vectors

    LB = zeros(numel(startVec), 1);
    UB = zeros(numel(startVec), 1);

    LB(1:n_frames-2)            = 0;                UB(1:n_frames-2)            = pi;      % local curvature c (in rad)
    LB(n_frames-1:2*n_frames-3) = 0;                UB(n_frames-1:2*n_frames-3) = 3;       % local distances d (in d')
    LB(2*n_frames-2:end)        = -100;             UB(2*n_frames-2:end)        = 100;     % accelerations a

    paramEst = fmincon(objFun, startVec, [], [], [], [], LB, UB, [], options);

    [NLL, x_fit, probCor, c] = giveNLL(paramEst, S{irun});
    
    % call likelihood function again with paramEst to visualize discrimination matrix
end

%% Plotting
subplot(2, 3, 1)
[~, s_true] = pca(S{1}.x');
plot(s_true(:, 1), s_true(:, 2), 'ko-', 'markersize', 12, 'markerfacecolor', [1 0 0], 'linewidth', 1)
title('Ground truth trajectory')
hold on, box off, axis square, axis equal

subplot(2, 3, 2)
imagesc(S{1}.Pc_reshaped);
hold on, axis square, box off
colormap('gray'); 
cBar = colorbar; 
title(cBar,'Proportion correct'); 
cBar.Ticks = [0.35,1];
xticks([1 n_frames]); yticks([1 n_frames]);
xlim([1 n_frames]); ylim([1 n_frames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Simulated discriminability');
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);

subplot(2, 3, 4)
[~, s_fit] = pca(x_fit');
plot(s_fit(:, 1), s_fit(:, 2), 'ko-', 'markersize', 12, 'markerfacecolor', [0 1 0], 'linewidth', 1)
title('Estimated trajectory')
hold on, box off, axis square, axis equal

subplot(2, 3, 5)
imagesc(probCor);
hold on, axis square, box off
colormap('gray'); 
cBar = colorbar; 
title(cBar,'Predicted proportion correct'); 
cBar.Ticks = [0.35,1];
xticks([1 n_frames]); yticks([1 n_frames]);
xlim([1 n_frames]); ylim([1 n_frames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Predicted discriminability');
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);

subplot(2, 3, 6)
plot(probCor(:), S{1}.Pc_reshaped(:), 'ro')
hold on;
plot([.5 1], [.5 1], 'k--')
xlabel('Predicted performance')
ylabel('Observed performance')


