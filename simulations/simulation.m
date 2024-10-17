% This script simulates the behavioral responses of the AxB task in the 
% temporal straightening paper (2019)
close all; clear all; clc;

%% global constants
ExpParam.numFrames  = 5;
ExpParam.avgLocCurv = 50;
ExpParam.numTrials  = 1000; 
ExpParam.numDim     = 4; %dimensionality of the perceptual space

%% generate a perceptual trajectory of N frames
v_0              = ones(1,ExpParam.numDim)'; % first vector at t0
ExpParam.vectors = zeros(ExpParam.numDim, ExpParam.numFrames); 
unit             = @(vec) vec/sqrt(sum(vec.^2));

% define a length range for all vectors and assign a random length to each
% vector
min_length = 1; max_length = 1.5; %longer length = better performance
lengths    = (max_length - min_length) * rand(1,ExpParam.numFrames) + min_length;

% local curvature is sampled from a normal dist centered on the avgLocCurv
sigma_curv            = 10;
ExpParam.thetas       = abs(deg2rad(normrnd(ExpParam.avgLocCurv,sigma_curv,[ExpParam.numFrames,1])));

% randomly assign the direction of local curvature (clockwise or counterclockwise)
cw_ccw_order = randi([0, 1], 1, ExpParam.numFrames);

% generate the vectors
for i = 1:ExpParam.numFrames
    % local curvature
    theta = ExpParam.thetas(i);
    % define the rotation plane
    rotPlane = randsample(ExpParam.numDim, 2);
    % get the cw/ccw rotation matrices
    [cwR,ccwR] = rotmat(ExpParam.numDim,theta,rotPlane);
    % calculate the next vector by rotating the previous vector and moving
    % it to the head of the previous vector
    if i == 1
        ExpParam.vectors(:,i) = lengths(i) * unit(v_0);
    else
        if cw_ccw_order(i) == 1
            ExpParam.vectors(:,i) = lengths(i) * unit(cwR * ExpParam.vectors(:,i-1))...
                + ExpParam.vectors(:,i-1);
        else
            ExpParam.vectors(:,i) = lengths(i) * unit(ccwR * ExpParam.vectors(:,i-1))...
                + ExpParam.vectors(:,i-1);
        end
    end
end

%% run PCA for dimensionality reduction
pc_coeff = pca(ExpParam.vectors);
pc_coeff = pc_coeff(:, 1:2)'; % use only first two PCs for plotting

figure(1);
for i = 2:ExpParam.numFrames  
    plot([pc_coeff(1,i-1),pc_coeff(1,i)],[pc_coeff(2,i-1),pc_coeff(2,i)],'-o','LineWidth', 1.5)
    hold on;
end
axis equal; grid on; hold on;
title('Simulated perceptual trajectory in PC space');

%% visualize the perceptual trajectory for 2d
figure(1);
for i = 2:ExpParam.numFrames  
    plot([ExpParam.vectors(1,i-1),ExpParam.vectors(1,i)],[ExpParam.vectors(2,i-1),ExpParam.vectors(2,i)],'-o','LineWidth', 1.5)
    hold on;
end
axis equal; grid on; hold on;
xlim([round(min(min(ExpParam.vectors)))-3, round(max(max(ExpParam.vectors)))+3]); 
ylim([round(min(min(ExpParam.vectors)))-3, round(max(max(ExpParam.vectors)))+3]);
title('Simulated perceptual trajectory');

%% visualize the perceptual trajectory for >= 3d (plotting first 3 dimensions)
% figure(1);
% for i = 2:ExpParam.numFrames  
% plot3([ExpParam.vectors(1,i-1), ExpParam.vectors(1,i)], [ExpParam.vectors(2,i-1), ExpParam.vectors(2,i)],...
%     [ExpParam.vectors(3,i-1), ExpParam.vectors(3,i)], '-o', 'LineWidth', 1.5)    
% hold on;
% end
% grid on; hold on;
% xlim([min(ExpParam.vectors(1, :)) - 3, max(ExpParam.vectors(1, :)) + 3]);
% ylim([min(ExpParam.vectors(2, :)) - 3, max(ExpParam.vectors(2, :)) + 3]);
% zlim([min(ExpParam.vectors(3, :)) - 3, max(ExpParam.vectors(3, :)) + 3]);
% axis equal;
% title('Simulated perceptual trajectory in 3D');

%% generate n-d isotropic gaussians centered on each frame (visualization ONLY, 2d)
% mus = ExpParam.vectors;
% sigma = 1;
% 
% % Create a grid in each dimension
% grids = cell(1, ExpParam.numDim);
% for dim = 1:ExpParam.numDim
%     grids{dim} = (round(min(min(mus)))-3):0.1:(round(max(max(mus)))+3);
% end
% 
% % Create a mesh grid for the specified number of dimensions
% grid_dims = cell(1, ExpParam.numDim);
% [grid_dims{1:ExpParam.numDim}] = ndgrid(grids{:}); % for 3d: [x,y,z]
% 
% grid_vals = NaN(length(grid_dims{1})^ExpParam.numDim,ExpParam.numDim);
% for i = 1:ExpParam.numDim
%     grid_vals(:,i) = grid_dims{i}(:);
% end
% 
% % Calculate Gaussian values at each point on the grid
% for i = 1:ExpParam.numFrames
%     gaussians{i} = mvnpdf(grid_vals, mus(:, i)', eye(ExpParam.numDim) * sigma);
%     % Reshape the result to match the grid size
%     gaussians{i} = reshape(gaussians{i}, size(grid_dims{1}));
% end
% 
% % visualize the gaussians in 2d 
% % combine the gaussians for visualization
% combined_gaussians = zeros(size(gaussians{1}));
% for i = 1:ExpParam.numFrames
%     combined_gaussians = combined_gaussians + gaussians{i};
% end
% 
% x = grid_dims{1};
% y = grid_dims{2};
% 
% figure(2);
% contourf(x,y,combined_gaussians(:,:,1),20,'LineStyle','none');
% colorbar; axis equal; 


%% simulate the AXB responses
% get all pairwise combinations (A,B), X will be identical to A or B
ExpParam.all_pairs = [repelem(1:ExpParam.numFrames,ExpParam.numFrames)',...
    repmat(1:ExpParam.numFrames,1,ExpParam.numFrames)'];

% simulate responses 
mus       = ExpParam.vectors;
sigma     = 1;
sigma_mat = eye(ExpParam.numDim) * sigma;
distance  = @(p1,p2) sqrt(sum((p1-p2).^2));
Data.resp_mat  = NaN(ExpParam.numTrials,length(ExpParam.all_pairs)); 

for i = 1:length(ExpParam.all_pairs)
    for j = 1:ExpParam.numTrials
        simA = mvnrnd(mus(:,ExpParam.all_pairs(i,1)),sigma_mat,1);
        simB = mvnrnd(mus(:,ExpParam.all_pairs(i,2)),sigma_mat,1);
        if rem(j,2) == 0
            simX = mvnrnd(mus(:,ExpParam.all_pairs(i,1)),sigma_mat,1); %draw from A
            dist_AX = distance(simA,simX);
            dist_BX = distance(simB,simX);
                if dist_AX < dist_BX 
                    Data.resp_mat(j,i) = 1; %correct response
                else
                    Data.resp_mat(j,i) = 0; %incorrect response
                end
        else
            simX = mvnrnd(mus(:,ExpParam.all_pairs(i,2)),sigma_mat,1); %draw from B
            dist_AX = distance(simA,simX);
            dist_BX = distance(simB,simX);
                if dist_BX < dist_AX  
                    Data.resp_mat(j,i) = 1; %correct response
                else
                    Data.resp_mat(j,i) = 0; %incorrect response
                end
        end 
    end
end
        
% calculate proportion correct
Pc = NaN(1,length(ExpParam.all_pairs));
for i = 1:length(ExpParam.all_pairs)
    Pc(i) = sum(Data.resp_mat(:,i))/ExpParam.numTrials;
end
Pc_reshaped = reshape(Pc,ExpParam.numFrames,ExpParam.numFrames);

%% visualize the simulated data
figure(3);
imagesc(Pc_reshaped);
colormap('gray'); 
c = colorbar; 
title(c,'Proportion correct'); 
c.Ticks = [0.5,1];
xticks([1 ExpParam.numFrames]); yticks([1 ExpParam.numFrames]);
xlim([1 ExpParam.numFrames]); ylim([1 ExpParam.numFrames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Simulated discriminability, %d trials', ExpParam.numTrials);
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);
axis equal; 

%% save data
cd '/Users/nguyentiendung/GitHub/perceptual-straightening/data/simulations'
save('ExpParam.mat', 'ExpParam');
save('Data.mat', 'Data');
save('discriminality_matrix.mat', 'Pc_reshaped');

% print average local curvature
disp(['Average local curvature: ' num2str(rad2deg(mean(ExpParam.thetas))) ' degrees'])
