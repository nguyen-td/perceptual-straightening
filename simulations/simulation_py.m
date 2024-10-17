% Function to generate simulated trajectories from behavioral responses. 
% Is intended to be called within Python but can also be called as
% standalone. Based on simulation.m without the plotting.
%
% Inputs:
%   avg_curvature - average curvature in degrees
%   n_frames      - number of frames/nodes
%   n_dim         - number of dimensions
%   n_trials      - number of trials
%
% Outputs:
%   ExpParam      - Structure containing the experimental parameters
%   Data          - Structure containing the trial matrix with correct (1)
%                   and incorrect (0) responses
%   Pc_reshaped   - (n_frames x n_frames) discriminability matrix

function [ExpParam, Data, Pc_reshaped] = simulation_py(avg_curvature, n_frames, n_dim, n_trials)
    % global constants
    ExpParam.numFrames  = n_frames;
    ExpParam.avgLocCurv = avg_curvature;
    ExpParam.numTrials  = n_trials; 
    ExpParam.numDim     = n_dim; %dimensionality of the perceptual space

    % generate a perceptual trajectory of N frames
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

    % simulate the AXB responses
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

end