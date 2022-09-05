% function [s,c]=mhg([M,K,lambda],alpha,p,q,x,[y])
%
% computes the truncated hypergeometric function pFq ^alpha(p;q;x;y) 
% of one OR two matrix arguments
%
% K, lambda, and y are optional, but if lambda is specfified, so must K
%
% The sum is over |kappa|<=M and 
%   a) if K is specified, also restricted over kappa_1<=K
%   b) if lambda is specified, also restricted over kappa_i<=lambda_i for
%      all i
% p,q,x,y are arrays, so hg(6,9,[3 4],[5 6 7],[1,2],[8 9]) is 
% 2F3^9([3 4];[5 6 7];[1 2];[8,9]) with the sum over all partitions
% |kappa|<=6
% 
% If y is omitted, the hypergeometric function of 1 matrix argument
%                    is computed.
% 
% c is a vector of size M+1 for the marginal sums for partions of sizes
% 0, ..., M
%
% alpha is a parameter, whose value is important and most often is 1 or 2:
% a) If the hypergeometric series is a sum of ZONAL POLYNOMIALS, then alpha=2.
%    We believe this to be the case most often encountered in practice.
% b) If the hypergeometric series is a sum of SCHUR FUNCTIONS, then alpha=1. 
% 
% Plamen Koev
% Department of Mathematics
% Massachusetts Institute of Technology
%
% Copyright (c) 2004-2007 Plamen Koev. See COPYRIGHT.TXT for details
%
% Reference: Plamen Koev and Alan Edelman, The Efficient Evaluation of the 
% Hypergeometric Function of a Matrix Argument, Math. Comp. 75 (2006), 
% 833-846.

error('No mex file found for this platform. Run "mex mhg.c" at the MATLAB prompt.');                    
