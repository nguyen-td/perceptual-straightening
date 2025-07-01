% Small function to compute pixel-domain curvature.

%% Setup
clear all
close all
clc

addpath(genpath('/Users/tn22693/GitHub/perceptual-straightening/'))

%% Load video
subject = 'ryan';
category = 'natural';
eccentricity = 'periphery';
movie_id = 6;
diameter = 24; % 6, 24, 36
movie_name = 'carnegie-dam';

v_folder = fullfile('data', 'yoon_stimulus_revised', ['diameter_' num2str(diameter,'%02.f') '_deg'], ['movie' num2str(movie_id,'%02.f') '-' movie_name]);
% v_folder = fullfile('data', 'yoon_stimulus_orig', ['movie' num2str(movie_id,'%02.f') '-' movie_name]);
files = dir(v_folder);

iframe = 1;
for ifile = 1:length(files)
    if contains(files(ifile).name, category)
       im{iframe} = imread(fullfile(v_folder, files(ifile).name)); 
       iframe = iframe + 1;
    end
end

I = double(cat(3, im{:})) / 255;

for iframe = 1:size(I, 3)
    imshow(I(:, :, iframe)); colormap('gray'), hold on, axis off, axis square 
    set(gca, 'YDir','normal')
    pause(.1)
end

%% Compute pixel curvature
n_frames = size(I, 3);
    
% compute local trajectory
v = diff(I, 1, 3);
v_hat = zeros(size(v, 1) * size(v, 2), size(v, 3));

for iframe = 1:size(v, 3)
    v_t = v(:, :, iframe);
    v_hat(:, iframe) = v_t(:) / norm(v_t(:));
end

c = zeros(n_frames - 2, 1);
for iframe = 1:n_frames - 2
    c(iframe) = acos(dot(v_hat(:, iframe), v_hat(:, iframe+1)));
end

disp(rad2deg(mean(c)))