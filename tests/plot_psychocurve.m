% Plot psychometric curve

%% Setup
clear all
close all
clc

addpath(genpath('/Users/tn22693/GitHub/perceptual-straightening/'))

%% Load data
dat_name = 'sim_0125.mat';
load(['data/sim/' dat_name])

%% Plot psychometric function
n_frames = size(S{1}.Pc_reshaped, 1);
rep_idx = 1; 
% sim_0002, rep 6: 73%, 75% at 3 frames
% sim_0004, rep 1: 75%, 75% at 3 frames
% sim_0005, rep 5: 68%, 75% at 3.5 frames
% sim_0007, rep 1: 60%, 75% at 6.7 frames
% sim_0030, rep 1: 59%, 75% at 8.8 frames
% sim_0031, rep 1: 75%, 75% at 2.2 frames (very beautiful psychometric function)
% sim_0121, rep 1: 92%, 75% at < 1  frames, 20° perceptual curvature (very very easy task, very steep psychometric curve)
% sim_0121, rep 1: 92%, 75% at < 1  frames, 20° perceptual curvature (very very easy task, very steep psychometric curve)

% compute all possible frame differences
diffs = [];
props = [];

for i = 1:n_frames
    for j = 1:n_frames
        if i ~= j  % ignore diagonal (A == B)
            diffs(end+1) = abs(i - j);
            props(end+1) = S{5}.Pc_reshaped(i,j);
        end
    end
end

% average proportion correct for each unique difference
unique_diffs = unique(diffs);
mean_props = arrayfun(@(d) mean(props(diffs == d), 'omitnan'), unique_diffs);

% plot
figure;
plot(unique_diffs, mean_props, 'o-', 'LineWidth', 1.5);
xlabel('Frame difference (|A - B|)');
ylabel('Proportion correct');
title('Psychometric function');
axis square;
grid on;

disp(['Average proportion correct: ' num2str(mean(S{rep_idx}.Pc_reshaped, 'all'))])
