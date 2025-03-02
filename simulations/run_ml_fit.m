%% Setup
clearvars;
clear all
close all
clc

addpath(genpath('/Users/nguyentiendung/GitHub/perceptual-straightening/'))
load('sim_0134.mat')

%% Run fitting
[x_fit, probCor, c] = ml_fit(S);

%% Plotting
n_frames = size(S{1}.Pc_reshaped,1);

subplot(2, 3, 1)
s_true = pca(S{1}.x);
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

subplot(2, 3, 3)
plot(rad2deg(c), rad2deg(S{1}.c), 'go')
hold on;
plot([0 180], [0 180], 'k--')
xlabel('Estimated curvature')
ylabel('Ground truth curvature')

subplot(2, 3, 4)
s_fit = pca(x_fit);
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


