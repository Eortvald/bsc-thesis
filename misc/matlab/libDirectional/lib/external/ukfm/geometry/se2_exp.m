function [chi] = se2_exp(xi)
%SE2_EXP exponential
%
% Syntax:  [chi] = se2_exp(xi)
%
% Inputs:
%    phi - vector
%
% Outputs:
%    chi - matrix

chi = [so2_exp(xi(1)) so2_left_jacobian(xi(1))*xi(2:3);
    0 0 1];
end

