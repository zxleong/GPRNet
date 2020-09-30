function A2 = gridinterp(A,x,z,x2,z2,method)
% gridinterp.m
% 
% This function interpolates the electrical property matrix A (having row and column position
% vectors x and z, respectively) to form the matrix A2 (having row and column position vectors
% x2 and z2, respectively).  The interpolation method by default is nearest neighbour ('nearest').  
% Otherwise the method can be specified as 'linear', 'cubic', or 'spline'.
%
% Syntax:  A2 = gridinterp(A,x,z,x2,z2,method)
%
% by James Irving
% July 2005

if nargin==5; method = 'nearest'; end

% transpose for interpolation
A = A';

% transform position vectors into matrices for interp2
[x,z] = meshgrid(x,z);
[x2,z2] = meshgrid(x2,z2);

% perform the interpolation
A2 = interp2(x,z,A,x2,z2,method);

% transpose back for output
A2 = A2';
