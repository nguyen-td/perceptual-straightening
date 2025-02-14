% Function to orthogonalize the acceleration w.r.t. the previous
% displacement vector, s.t. a(t) ⊥ v(t-1).
%
% Inputs:
%   a     - [n_dim] acceleration vector
%   v_hat - [n_dim] displacement vector
%
% Output:
%   a_hat_orth - [n_dim] orthogonal displacement vector 

function a_hat_orth = orthogonalize_acc(a, v_hat)
    proj_a2v = ((a' * v_hat) / (v_hat' * v_hat)) * v_hat;
    a_hat_orth = a - proj_a2v;
    a_hat_orth = a_hat_orth / norm(a_hat_orth);
    assert(abs(v_hat' * a_hat_orth) <= 1e-6)
end