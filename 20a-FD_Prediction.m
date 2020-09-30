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

ep_stack = struct2array(load('Field/Data/prediction/ep_FieldPrediction.mat'))';
ep_stack(1:10,:) = 1; %reaffirm air layer

%FDTD only take odd rows and column, so im dropping the last column
Stacked_1D = nan(2560,207);

for num=1:207


    width = 21;
    ep_pick = ep_stack(:,num)';
    ep =  repmat(ep_pick,width,1); %dielectric permitivity

    mu = ones(width,numel(ep_pick));

    sig = ones(width,numel(ep_pick)) * 0.001;


    %Specify depth layer
    z = -0.50:0.05:(numel(ep_pick)-11)*0.05;

    %Specify x distance
%     x_dist = 2*0.3048; %source interval is 2ft
    x_dist = 0.05; %source interval is 2ft

    x = 0:x_dist:x_dist*(width-1);

    %Set dx and dz
%     dx = 0.03048;
    dx = 0.025;
    dz = 0.025;

    %make source pulse
    dt = 1e-10;
    nt=2560;
    tt = (nt-1)*dt;
    t=0:dt:tt;
    fc = 100e6;
    srcpulse = -ricker(fc,nt,dt);   

    % interpolate electrical property grids to proper spatial discretization
    x2 = min(x):dx:max(x);
    z2 = min(z):dz:max(z);
    ep2 = gridinterp(ep,x,z,x2,z2,'nearest');
    mu2 = gridinterp(mu,x,z,x2,z2,'nearest');
    sig2 = gridinterp(sig,x,z,x2,z2,'nearest');


    % pad electrical property matrices for PML absorbing boundaries
    npml = 10;  % number of PML boundary layers
    [ep3,x3,z3] = padgrid(ep2,x2,z2,2*npml+1);
    [mu3,x3,z3] = padgrid(mu2,x2,z2,2*npml+1);
    [sig3,x3,z3] = padgrid(sig2,x2,z2,2*npml+1);


%     srcx = x';
    srcx = x(round(numel(x)/2));
    srcz = -0.05*ones(size(srcx));
    recx = srcx ;
    % recx = ((3:2:(206*2)+3)*0.3048)';
    recz = srcz;    
    srcloc = [srcx srcz];
    recloc = [recx recz];

    % set some output and plotting parameters
    outstep = 1;
    plotopt = [0 50 0.0002];

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

    Stacked_1D(:,num)=gather;
end

save('Field/Data/prediction/rawgather_Stacked1D_fieldprediction.mat','Stacked_1D');




