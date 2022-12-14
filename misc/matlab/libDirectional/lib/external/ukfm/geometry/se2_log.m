function [xi] = se2_log(chi)
%SE2_LOG logarithm
%
% Syntax:  [xi] = se2_log(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    xi - vector

phi = so2_log(chi(1:2, 1:2));
xi = [phi;
     so2_inv_left_jacobian(phi)*chi(1:2, 3)];
end
