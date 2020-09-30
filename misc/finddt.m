function dtmax = finddt(epmin,mumin,dx,dz);
% finddt.m
%
% This function finds the maximum time step that can be used in the 2-D
% FDTD modeling codes TM_model2d.m and TE_model2d.m, such that they remain
% numerically stable.  Second-order-accurate time and fourth-order-accurate 
% spatial derivatives are assumed (i.e., O(2,4)).
%
% Syntax: dtmax = finddt(epmin,mumin,dx,dz)
%
% where dtmax = maximum time step for FDTD to be stable
%       epmin = minimum relative dielectric permittivity in grid
%       mumin = minimum relative magnetic permeability in grid
%       dx = spatial discretization in x-direction (m)
%       dz = spatial discretization in z-direction (m)
%
% by James Irving
% July 2005

% convert relative permittivity and permeability to true values
mu0 = 1.2566370614e-6;
ep0 = 8.8541878176e-12;
epmin = epmin*ep0;
mumin = mumin*mu0;

% determine maximum allowable time step for numerical stability
dtmax = 6/7*sqrt(epmin*mumin/(1/dx^2 + 1/dz^2));