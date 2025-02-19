function [NLL, x, probCor, c] = giveNLL(paramVec, S)
    
    % Step 1: Unpack the data
    nObsCor = S.Pc_reshaped .* S.num_trials_mat; 
    nObsTotal = S.num_trials_mat;
    n_frames = size(S.Pc_reshaped,1);
    n_dim = size(S.x, 1);

    % Step 2: Unpack the parameter vector
    c = paramVec(1:n_frames-2);
    d = paramVec(n_frames-1:2*n_frames-3);
    a = reshape(paramVec(2*n_frames-2:end), n_dim, n_frames-2);

    % Step 3: derive model prediction for this experiment
    v_hat = zeros(n_dim, n_frames);
    v_hat(1,1) = 1;
    [x, v, c, a] = compute_trajectory(d, c, a, v_hat, n_frames, n_dim);

    % remove for-loop
    % Get perceptual distances
    distance    = @(p1,p2) sqrt(sum((p1-p2).^2));
    for ir = 1:n_frames
        for ic = 1:n_frames
            d(ir, ic) = distance(x(:,ir), x(:, ic));
        end
    end
    
    % Get predicted probability correct
    probCor = normcdf(d./sqrt(2)) .* normcdf(d ./ 2) + normcdf(-d./sqrt(2)) .* normcdf(-d ./ 2);
        
    % Get LLH
    NLL = -sum(sum((nObsCor .* log(probCor)) + (nObsTotal - nObsCor) .* log(1 - probCor)));  
end