% This script simulates the behavioral responses of the AxB task in the 
% temporal straightening paper (2019)
close all; clear all; clc;

%% global constants
ExpParam.numFrames  = 11;
ExpParam.numReps    = 10; 
ExpParam.numDim     = 6; % dimensionality of the perceptual space

d_mu = 1; 
d_sigma = 0.75;

c_mu = 45;
c_sigma = 15;

a_mu = 0;
a_sigma = 2; % needs to be larger than zero

%% generate a perceptual trajectory of N frames
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


%% simulate the AXB responses
% get all pairwise combinations (A,B), X will be identical to A or B
ExpParam.all_pairs = [repelem(1:ExpParam.numFrames,ExpParam.numFrames)',...
    repmat(1:ExpParam.numFrames,1,ExpParam.numFrames)'];

% simulate responses 
mus       = x;
sigma     = 1;
sigma_mat = eye(ExpParam.numDim) * sigma;
distance  = @(p1,p2) sqrt(sum((p1-p2).^2));
Data.resp_mat  = NaN(ExpParam.numReps,length(ExpParam.all_pairs)); 

% add 10% abort chance
% save this as a structure
% 

for i = 1:length(ExpParam.all_pairs)
    for j = 1:ExpParam.numReps
        simA = mvnrnd(mus(:,ExpParam.all_pairs(i,1)),sigma_mat,1); % frame A
        simB = mvnrnd(mus(:,ExpParam.all_pairs(i,2)),sigma_mat,1); % frame B
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
    Pc(i) = sum(Data.resp_mat(:,i))/ExpParam.numReps;
end
Pc_reshaped = reshape(Pc,ExpParam.numFrames,ExpParam.numFrames);

%% visualize the simulated data
subplot(1,3,1)
plot(x(1, :), x(2, :), 'ko-', 'markersize', 12, 'markerfacecolor', [1 0 0], 'linewidth', 1)
hold on, box off, axis equal, axis square

subplot(1,3,2)
plot(ExpParam.c, c, 'r+', 'linewidth', 1)
hold on, box off, axis square
plot([0 pi], [0 pi], 'k--')
plot(mean(ExpParam.c), mean(c), 'ko', 'MarkerSize', 12, 'markerfacecolor', [0 .5 1])


subplot(1,3,3)
imagesc(Pc_reshaped);\
hold on, axis square, box off
colormap('gray'); 
cBar = colorbar; 
title(cBar,'Proportion correct'); 
cBar.Ticks = [0.35,1];
xticks([1 ExpParam.numFrames]); yticks([1 ExpParam.numFrames]);
xlim([1 ExpParam.numFrames]); ylim([1 ExpParam.numFrames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Simulated discriminability, %d trials', ExpParam.numReps);
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);

intention = c_mu
reality = rad2deg(mean(c))
performance = mean(Pc)

% %% save data
% cd '/Users/nguyentiendung/GitHub/perceptual-straightening/data/simulations'
% save('ExpParam.mat', 'ExpParam');
% save('Data.mat', 'Data');
% save('discriminality_matrix.mat', 'Pc_reshaped');
% 
% % print average local curvature
% disp(['Average local curvature: ' num2str(rad2deg(mean(ExpParam.thetas))) ' degrees'])
