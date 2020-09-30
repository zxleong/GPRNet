% Adapted from:
%
% Irving, J., & Knight, R. (2006). Numerical modeling of ground-penetrating 
% radar in 2-D using MATLAB. Computers & Geosciences, 32(9), 1247–1258. 
% https://doi.org/10.1016/j.cageo.2005.11.006
%
% (a 2-D, FDTD, reflection GPR modeling code in MATLAB)
%
% by James Irving
% July 2005
clear all
addpath(genpath('./misc'));


%load true model

ep = struct2array(load('Synthetic/Data/2D/ep_ypred2D.mat'))';
ep = ep(:,1:end-1); %make sure rows and columns are odd number because FDTD only takes odd number


row_height = 111; %depth of dx=0.05
width = 251; %number of shots in x direction

mu = ones(width,row_height);
sig = ones(width,row_height) * 0.001;


%Specify depth layer
z = -0.50:0.05:(row_height-11)*0.05;

%Specify x distance
x_dist = 0.05;
x = 0:x_dist:x_dist*(width-1);


% set dx and dz here (m) using the above results as a guide
dx = 0.025;
dz = 0.025;


% (using the proper values of dt and tmax this time)

dt = 8e-11;
tt = (1280-1)*dt;
t=0:dt:tt;
fc = 120e6;
srcpulse = blackharrispulse(fc,t);   

% interpolate electrical property grids to proper spatial discretization
% NOTE:  we MUST use dx/2 here because we're dealing with electrical property matrices

x2 = min(x):dx:max(x);
z2 = min(z):dx:max(z);
ep2 = gridinterp(ep,x,z,x2,z2,'nearest');
mu2 = gridinterp(mu,x,z,x2,z2,'nearest');
sig2 = gridinterp(sig,x,z,x2,z2,'nearest');


% pad electrical property matrices for PML absorbing boundaries
npml = 10;  % number of PML boundary layers
[ep3,x3,z3] = padgrid(ep2,x2,z2,2*npml+1);
[mu3,x3,z3] = padgrid(mu2,x2,z2,2*npml+1);
[sig3,x3,z3] = padgrid(sig2,x2,z2,2*npml+1);

% clear unnecessary matrices taking up memory
% clear x x2 z z2 ep ep2 mu mu2 sig sig2 

% create source and receiver location matrices
% (rows are [x location (m), z location (m)])

srcx = x';
srcz = -0.05*ones(size(srcx));
recx = srcx ;
recz = srcz;    
srcloc = [srcx srcz];
recloc = [recx recz];

% set some output and plotting parameters
outstep = 1;
plotopt = [1 50 0.0002];

% pause
% disp('Press any key to begin simulation...');
% disp(' ');
% pause;
close all

% pml (1) or no pml (0) on top
top_pml=1;

% run the simulation
tic;
[gather,tout,srcx,srcz,recx,recz] = TM_model2d(ep3,mu3,sig3,x3,z3,srcloc,recloc,srcpulse,t,npml,top_pml,outstep,plotopt);
disp(' ');
disp(['Total running time = ',num2str(toc/3600),' hours']);

% extract common offset reflection GPR data from multi-offset data cube and plot the results
for i=1:length(srcx);
    co_data(:,i) = gather(:,i,i);
end

save('Synthetic/Data/2D/fdraw_predicted_data.mat','co_data');


