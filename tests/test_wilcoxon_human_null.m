% Compute Wilcoxon-signed rank statistic to compare differences between curvatures from human data and null data.

clear 
clc

%% Load data
f_name = 'JX_gray_mats.csv';
path_name = ['/Users/tn22693/Downloads/' f_name];

data = readmatrix(path_name);
p = ranksum(data(:, 1), data(:, 2));