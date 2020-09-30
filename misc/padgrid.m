function [A2,x2,z2] = padgrid(A,x,z,n)
% padgrid.m
% 
% This function pads the electrical property matrix A (having row and column position vectors
% x and z, respectively) with n elements around each side to create the matrix A2 (having row
% and column position vectors x2 and z2, respectively).  The properties in the padded regions
% are simply the properties of the original matrix extended outwards.
%
% Syntax:  [A2,x2,z2] = padgrid(A,x,z,n)
%
% by James Irving
% July 2005

% determine the new position vectors
dx = x(2)-x(1);
dz = z(2)-z(1);
x2 = (x(1)-n*dx):dx:(x(end)+n*dx+1e-10);
z2 = (z(1)-n*dz):dz:(z(end)+n*dz+1e-10);

% pad the grid
A2 = [repmat(A(:,1),1,n), A, repmat(A(:,end),1,n)];
A2 = [repmat(A2(1,:),n,1); A2; repmat(A2(end,:),n,1)];
