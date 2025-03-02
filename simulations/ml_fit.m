% Maximum likelihood estimation of perceptual trajectories.

function [x_fit, probCor, c] = ml_fit(S)
    options  = optimset('Display', 'iter', 'MaxIter', 1000, 'MaxFuneval', 10^5);
    
    irun = 1;
    objFun   = @(paramVec) giveNLL(paramVec, S{irun});
    
    % define number of frames and number of dimensions
    n_frames = size(S{irun}.Pc_reshaped,1);
    n_dim = size(S{irun}.x,1);
    
    % initialize start vector
    startVec = zeros(n_frames-2 + n_frames-1 + n_dim * (n_frames-2), 1);
    startVec(1:n_frames-2) = abs(deg2rad(normrnd(60, 10, [n_frames-2, 1]))); % c
    startVec(n_frames-1:2*n_frames-3) = abs(normrnd(1, 0.5, [n_frames-1, 1])); % d
    startVec(2*n_frames-2:end) = normrnd(0, 2, [n_dim, n_frames-2]); % a
    % startVec(1:n_frames-2) = zeros(n_frames-2, 1) + deg2rad(60); % c
    % startVec(n_frames-1:2*n_frames-3) = ones(n_frames-1, 1); % d
    % startVec(2*n_frames-2:end) = ones(n_dim, n_frames-2); % a

    % paramVec(1:N-2) = local curvature, starting at frame 1
    % paramVec(N-1:2N-3) = local displacement vector distance
    % paramVec(2N-2:end) = acceleration vectors

    LB = zeros(numel(startVec), 1);
    UB = zeros(numel(startVec), 1);

    LB(1:n_frames-2)            = 0;                UB(1:n_frames-2)            = pi;      % local curvature c (in rad)
    LB(n_frames-1:2*n_frames-3) = 0;                UB(n_frames-1:2*n_frames-3) = 3;       % local distances d (in d')
    LB(2*n_frames-2:end)        = -100;             UB(2*n_frames-2:end)        = 100;     % accelerations a

    paramEst = fmincon(objFun, startVec, [], [], [], [], LB, UB, [], options);
    % paramEst = bads(objFun, startVec', LB', UB', LB', UB');

    [NLL, x_fit, probCor, c] = giveNLL(paramEst, S{irun});
end

