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


ep_stack = struct2array(load('Field/data/v3/ep.mat'));

gather_stack = NaN(50000,8000);
tic;
for num=1:50000
    ep_pick = ep_stack(num,:); 
    
    row_height = numel(ep_pick); %depth of dx=0.05
    width = 21; %number of shots in x direction

    
    ep =  repmat(ep_pick,width,1);
    mu = ones(width,row_height);
    sig = ones(width,row_height) * 0.001;
        
    %Specify depth layer
    z = -0.50:0.05:(row_height-11)*0.05;


    %Specify x distance
    x_dist = 0.05;
    x = 0:x_dist:x_dist*(width-1);
    
    %Set dx and dz
    dx = 0.025;
    dz = 0.025;

    %make source pulse
    scale=20;
    ns_t = 400;
    dt = 1e-9/scale;
    nt = ns_t*scale;
    tt = (nt-1)*dt;
    t=0:dt:tt;
    fc = 100e6;
    srcpulse = -ricker(fc,nt,dt);   

    % interpolate electrical property grids to proper spatial discretization
    % NOTE:  we MUST use dx/2 here because we're dealing with electrical property matrices
%     disp('Interpolating electrical property matrices...');
%     disp(' ');
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



    % create source and receiver location matrices
    % (rows are [x location (m), z location (m)])
    srcx = x(round(numel(x)/2)); %middle
    srcz = -0.05*ones(size(srcx)); %source at 5mm above ground
    
    
    recx = srcx ;
    recz = srcz;    
    srcloc = [srcx srcz];
    recloc = [recx recz];
    
    % set some output and plotting parameters
    outstep = 1;
    plotopt = [0 50 0.0002];

    % pml (1) or no pml (0) on top
    top_pml=1;

    % run the simulation
    tic;
    disp(' ');
    disp(['Simulating number ', num2str(num), ' th simulation...'] );
    
    [gather,tout,srcx,srcz,recx,recz] = TM_model2d(ep3,mu3,sig3,x3,z3,srcloc,recloc,srcpulse,t,npml,top_pml,outstep,plotopt);
    disp(' ');
    disp(['Sumlation running time = ',num2str(toc/3600),' hours']);
    


    % inserting gather into empty stack
    gather_stack(num,:) = gather;
    save('Field/Data/fdrawgathers.mat','gather_stack');
    

end





