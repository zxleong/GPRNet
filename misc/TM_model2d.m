function [gather,tout,srcx,srcz,recx,recz] = TM_model2d(ep,mu,sig,xprop,zprop,srcloc,recloc,srcpulse,t,npml,top_pml,outstep,plotopt)
% TM_model2d.m
% 
% This is a 2-D, TM-mode, FDTD modeling program for reflection ground-penetrating radar.  
% See TM_run_example.m for an example of how to use this code.
%
% Features:
% - convolutional PML absorbing boundaries (Roden and Gedney, 2000)
% - calculations in PML boundary regions have been separated from main modeling region for speed
% - second-order accurate time derivatives, fourth-order-accurate spatial derivatives (O(2,4))
%
% Syntax:  [gather,tout,srcx,srcz,recx,recz] = TM_model2d(ep,mu,sig,xprop,zprop,srcloc,recloc,srcpulse,t,npml,outstep,plotopt)
%
% where gather = output data cube (common source gathers are stacked along the third index)
%       tout = output time vector corresponding to first index in gather matrix (s)
%       srcx = vector containing actual source x locations after discretization (m)
%       srcz = vector containing actual source z locations after discretization (m)
%       recx = vector containing actual receiver x locations after discretization (m)
%       recz = vector containing actual receiver z locations after discretization (m)
%       ep,mu,sig = electrical property matrices  (rows = x, columns = z;  need odd # of rows and columns; 
%             already padded for PML boundaries; ep and mu are relative to free space; sig units are S/m)
%       xprop,zprop = position vectors corresponding to electrical property matrices (m)
%       srcloc = matrix containing source locations (rows are [x (m), z (m)])
%       recloc = matrix containing receiver locations (format same as above)
%       srcpulse = source pulse vector (length must be equal to the number of iterations)
%       t = time vector corresponding to srcpulse (used for determining time step and number of iterations) (s)
%       npml = number of PML absorbing boundary cells
%       outstep = write a sample to the output matrix every so many iterations (default=1)
%       plotopt = plot Ey wavefield during simulation?  
%           (vector = [{0=no, 1=yes}, {output every # of iterations}, {colorbar threshold}])
%           (default = [1 50 0.05])
%
% Notes:
% - all matrices in this code need to be transposed before plotting, as the first index always refers 
%   to the horizontal direction, and the second index to the vertical direction
% - locations in space are referred to using x and z
% - indices in electric and magnetic field matrices are referred to using (i,j)  [i=horizontal,j=vertical]
% - indices in electrical property matrices are referred to using (k,l)  [k=horizontal,l=vertical]
%
% by James Irving
% July 2005

% ------------------------------------------------------------------------------------------------
% SET DEFAULTS, AND TEST A FEW THINGS TO MAKE SURE ALL THE INPUTS ARE OK
% ------------------------------------------------------------------------------------------------

if nargin==10; outstep=1; plotopt=[1 50 0.05]; end
if nargin==11; plotopt=[1 50 0.05]; end

if size(mu)~=size(ep) | size(sig)~=size(ep); disp('ep, mu, and sig matrices must be the same size'); return; end
if [length(xprop),length(zprop)]~=size(ep); disp('xprop and zprop are inconsistent with ep, mu, and sig'); return; end
if mod(size(ep,1),2)~=1 | mod(size(ep,2),2)~=1; disp('ep, mu, and sig must have an odd # of rows and columns'); return; end
if size(srcloc,2)~=2 | size(recloc,2)~=2; disp('srcloc and recloc matrices must have 2 columns'); return; end
if max(srcloc(:,1))>max(xprop) | min(srcloc(:,1))<min(xprop) | max(srcloc(:,2))>max(zprop)...
        | min(srcloc(:,2))<min(zprop); disp('source vector out of range of modeling grid'); return; end 
if max(recloc(:,1))>max(xprop) | min(recloc(:,1))<min(xprop) | max(recloc(:,2))>max(zprop)...
        | min(recloc(:,2))<min(zprop); disp('receiver vector out of range of modeling grid'); return; end
if length(srcpulse)~=length(t); disp('srcpulse and t vectors must have same # of points'); return; end
if npml>=length(xprop)/2 | npml>=length(zprop)/2; disp('too many PML boundary layers for grid'); return; end
if length(plotopt)~=3; disp('plotopt must be a 3 component vector'); return; end


% ------------------------------------------------------------------------------------------------
% DETERMINE SOME INITIAL PARAMETERS
% ------------------------------------------------------------------------------------------------

% determine true permittivity and permeability matrices from supplied relative ones
ep0 = 8.8541878176e-12;             % dielectric permittivity of free space 
mu0 = 1.2566370614e-6;              % magnetic permeability of free space
ep = ep*ep0;                        % true permittivity matrix            
mu = mu*mu0;                        % true permeability matrix

% determine number of field nodes and discretization interval
nx = (length(xprop)+1)/2;           % maximum number of field nodes in the x-direction
nz = (length(zprop)+1)/2;           % maximum number of field nodes in the z-direction
dx = 2*(xprop(2)-xprop(1));         % electric and magnetic field spatial discretization in x (m)
dz = 2*(zprop(2)-zprop(1));         % electric and magnetic field spatial discretization in z (m)

% x and z position vectors corresponding to Hx, Hz, and Ey field matrices
% (these field matrices are staggered in both space and time, and thus have different coordinates)
xHx = xprop(2):dx:xprop(end-1);
zHx = zprop(1):dz:zprop(end);
xHz = xprop(1):dx:xprop(end);
zHz = zprop(2):dz:zprop(end-1);
xEy = xHx;
zEy = zHz;

% determine source and receiver (i,j) indices in Ey field matrix,
% and true coordinates of sources and receivers in numerical model (after discretization)
nsrc = size(srcloc,1);                          % number of sources
nrec = size(recloc,1);                          % number of receivers
for s=1:nsrc;                                   
    [temp,srci(s)] = min(abs(xEy-srcloc(s,1))); % source x index in Ey field matrix
    [temp,srcj(s)] = min(abs(zEy-srcloc(s,2))); % source z index in Ey field matrix
    srcx(s) = xEy(srci(s));                     % true source x location
    srcz(s) = zEy(srcj(s));                     % true source z location
end
for r=1:nrec;                                   
    [temp,reci(r)] = min(abs(xEy-recloc(r,1))); % receiver x index in Ey field matrix
    [temp,recj(r)] = min(abs(zEy-recloc(r,2))); % receiver z index in Ey field matrix
    recx(r) = xEy(reci(r));                     % true receiver x location
    recz(r) = zEy(recj(r));                     % true receiver z location
end

% determine time stepping parameters from supplied time vector
dt = t(2)-t(1);                                 % temporal discretization
numit = length(t);                              % number of iterations


% ------------------------------------------------------------------------------------------------
% COMPUTE FDTD UPDATE COEFFICIENTS FOR ENTIRE SIMULATION GRID
% note:  these matrices are twice as large as the field component matrices
% (i.e., the same size as the electrical property matrices)
% ------------------------------------------------------------------------------------------------

disp('Determining update coefficients for simulation region...')

% set the basic PML parameters, keeping the following in mind...   
% - maximum sigma_x and sigma_z vary in heterogeneous media to damp waves most effiently
% - Kmax = 1 is the original PML of Berenger (1994); Kmax > 1 can be used to damp evanescent waves
% - keep alpha = 0 except for highly elongated domains (see Roden and Gedney (2000))
m = 4;                                          % PML Exponent (should be between 3 and 4)
Kxmax = 5;                                      % maximum value for PML K_x parameter (must be >=1)
Kzmax = 5;                                      % maximum value for PML K_z parameter (must be >=1)
sigxmax = (m+1)./(150*pi*sqrt(ep./ep0)*dx);     % maximum value for PML sigma_x parameter
sigzmax = (m+1)./(150*pi*sqrt(ep./ep0)*dz);     % maximum value for PML sigma_z parameter
alpha = 0;                                      % alpha parameter for PML (CFS)

% indices corresponding to edges of PML regions in electrical property grids
kpmlLout = 1;                           % x index for outside of PML region on left-hand side
kpmlLin = 2*npml+2;                     % x index for inside of PML region on left-hand side
kpmlRin = length(xprop)-(2*npml+2)+1;   % x index for inside of PML region on right-hand side
kpmlRout = length(xprop);               % x index for outside of PML region on right-hand side
lpmlTout = 1;                           % z index for outside of PML region at the top
lpmlTin = 2*npml+2;                     % z index for inside of PML region at the top
lpmlBin = length(zprop)-(2*npml+2)+1;   % z index for inside of PML region at the bottom
lpmlBout = length(zprop);               % z index for outside of PML region at the bottom

% determine the ratio between the distance into the PML and the PML thickness in x and z directions
% done for each point in electrical property grids;  non-PML regions are set to zero
xdel = zeros(length(xprop),length(zprop));                      % initialize x direction matrix
k = kpmlLout:kpmlLin;  k = k(:);                                % left-hand PML layer
xdel(k,:) = repmat(((kpmlLin-k)./(2*npml)),1,length(zprop));
k = kpmlRin:kpmlRout; k = k(:);                                 % right-hand PML layer
xdel(k,:) = repmat(((k-kpmlRin)./(2*npml)),1,length(zprop));
zdel = zeros(length(xprop),length(zprop));                      % initialize z direction matrix
l = lpmlTout:lpmlTin;                                           % top PML layer
zdel(:,l) = repmat(((lpmlTin-l)./(2*npml)),length(xprop),1);
if top_pml==0
    zdel(:,l)=repmat(((lpmlTin-l)./(2*npml)),length(xprop),1)*0;
end
l = lpmlBin:lpmlBout;                                           % bottom PML layer
zdel(:,l) = repmat(((l-lpmlBin)./(2*npml)),length(xprop),1);

% determine PML parameters at each point in the simulation grid 
% (scaled to increase from the inside to the outside of the PML region)
% (interior non-PML nodes have sigx=sigz=0, Kx=Kz=1)
sigx = sigxmax.*xdel.^m;
sigz = sigzmax.*zdel.^m;
Kx = 1 + (Kxmax-1)*xdel.^m;
Kz = 1 + (Kzmax-1)*zdel.^m;

% determine FDTD update coefficients
Ca  = (1-dt*sig./(2*ep))./(1+dt*sig./(2*ep));
Cbx  = (dt./ep)./((1+dt*sig./(2*ep))*24*dx.*Kx);
Cbz  = (dt./ep)./((1+dt*sig./(2*ep))*24*dz.*Kz);
Cc = (dt./ep)./(1+dt*sig./(2*ep));
Dbx = (dt./(mu.*Kx*24*dx));
Dbz = (dt./(mu.*Kz*24*dz));
Dc = dt./mu;
Bx = exp(-(sigx./Kx + alpha)*(dt/ep0));
Bz = exp(-(sigz./Kz + alpha)*(dt/ep0));
Ax = (sigx./(sigx.*Kx + Kx.^2*alpha + 1e-20).*(Bx-1))./(24*dx);
Az = (sigz./(sigz.*Kz + Kz.^2*alpha + 1e-20).*(Bz-1))./(24*dz);

% clear unnecessary PML variables as they take up lots of memory
clear sigmax xdel zdel Kx Kz sigx sigz


% ------------------------------------------------------------------------------------------------
% RUN THE FDTD SIMULATION
% ------------------------------------------------------------------------------------------------

disp('Beginning FDTD simulation...')

% initialize gather matrix where data will be stored
gather = zeros(fix((numit-1)/outstep)+1,nrec,nsrc);

% loop over number of sources
for s=1:nsrc

    % zero all field matrices
    Ey = zeros(nx-1,nz-1);          % Ey component of electric field
    Hx = zeros(nx-1,nz);            % Hx component of magnetic field
    Hz = zeros(nx,nz-1);            % Hz component of magnetic field
    Eydiffx = zeros(nx,nz-1);       % difference for dEy/dx
    Eydiffz = zeros(nx-1,nz);       % difference for dEy/dz
    Hxdiffz = zeros(nx-1,nz-1);     % difference for dHx/dz
    Hzdiffx = zeros(nx-1,nz-1);     % difference for dHz/dx
    PEyx = zeros(nx-1,nz-1);        % psi_Eyx (for PML)
    PEyz = zeros(nx-1,nz-1);        % psi_Eyz (for PML)
    PHx = zeros(nx-1,nz);           % psi_Hx (for PML)
    PHz = zeros(nx,nz-1);           % psi_Hz (for PML)
    
    % time stepping loop
    for it=1:numit
        
        % update Hx component...
        
        % determine indices for entire, PML, and interior regions in Hx and property grids
        i = 2:nx-2;  j = 3:nz-2;                % indices for all components in Hx matrix to update
        k = 2*i;  l = 2*j-1;                    % corresponding indices in property grids
        kp = k((k<=kpmlLin | k>=kpmlRin));      % corresponding property indices in PML region
        lp = l((l<=lpmlTin | l>=lpmlBin));
        ki = k((k>kpmlLin & k<kpmlRin));        % corresponding property indices in interior (non-PML) region
        li = l((l>lpmlTin & l<lpmlBin));    
        ip = kp./2;  jp = (lp+1)./2;            % Hx indices in PML region
        ii = ki./2;  ji = (li+1)./2;            % Hx indices in interior (non-PML) region
        
        % update to be applied to the whole Hx grid
        Eydiffz(i,j) = -Ey(i,j+1) + 27*Ey(i,j) - 27*Ey(i,j-1) + Ey(i,j-2);
	    Hx(i,j) = Hx(i,j) - Dbz(k,l).*Eydiffz(i,j);
        
        % update to be applied only to the PML region
        PHx(ip,j) = Bz(kp,l).*PHx(ip,j) + Az(kp,l).*Eydiffz(ip,j);
        PHx(ii,jp) = Bz(ki,lp).*PHx(ii,jp) + Az(ki,lp).*Eydiffz(ii,jp);
        Hx(ip,j) = Hx(ip,j) - Dc(kp,l).*PHx(ip,j);
        Hx(ii,jp) = Hx(ii,jp) - Dc(ki,lp).*PHx(ii,jp);
        
    
        % update Hz component...

        % determine indices for entire, PML, and interior regions in Hz and property grids
        i = 3:nx-2;  j = 2:nz-2;                % indices for all components in Hz matrix to update
        k = 2*i-1;  l = 2*j;                    % corresponding indices in property grids
        kp = k((k<=kpmlLin | k>=kpmlRin));      % corresponding property indices in PML region
        lp = l((l<=lpmlTin | l>=lpmlBin));
        ki = k((k>kpmlLin & k<kpmlRin));        % corresponding property indices in interior (non-PML) region
        li = l((l>lpmlTin & l<lpmlBin));
        ip = (kp+1)./2;  jp = lp./2;            % Hz indices in PML region
        ii = (ki+1)./2;  ji = li./2;            % Hz indices in interior (non-PML) region

        % update to be applied to the whole Hz grid
        Eydiffx(i,j) = -Ey(i+1,j) + 27*Ey(i,j) - 27*Ey(i-1,j) + Ey(i-2,j);
	    Hz(i,j) = Hz(i,j) + Dbx(k,l).*Eydiffx(i,j);
        
        % update to be applied only to the PML region
        PHz(ip,j) = Bx(kp,l).*PHz(ip,j) + Ax(kp,l).*Eydiffx(ip,j);
        PHz(ii,jp) = Bx(ki,lp).*PHz(ii,jp) + Ax(ki,lp).*Eydiffx(ii,jp);
        Hz(ip,j) = Hz(ip,j) + Dc(kp,l).*PHz(ip,j);
        Hz(ii,jp) = Hz(ii,jp) + Dc(ki,lp).*PHz(ii,jp);
        
        
        % update Ey component...
        
        % determine indices for entire, PML, and interior regions in Ey and property grids
        i = 2:nx-2;  j = 2:nz-2;                % indices for all components in Ey matrix to update
        k = 2*i;  l = 2*j;                      % corresponding indices in property grids
        kp = k((k<=kpmlLin | k>=kpmlRin));      % corresponding property indices in PML region
        lp = l((l<=lpmlTin | l>=lpmlBin));
        ki = k((k>kpmlLin & k<kpmlRin));        % corresponding property indices in interior (non-PML) region
        li = l((l>lpmlTin & l<lpmlBin));
        ip = kp./2;  jp = lp./2;                % Ey indices in PML region
        ii = ki./2;  ji = li./2;                % Ey indices in interior (non-PML) region

        % update to be applied to the whole Ey grid
        Hxdiffz(i,j) = -Hx(i,j+2) + 27*Hx(i,j+1) - 27*Hx(i,j) + Hx(i,j-1);
        Hzdiffx(i,j) = -Hz(i+2,j) + 27*Hz(i+1,j) - 27*Hz(i,j) + Hz(i-1,j);
        Ey(i,j) = Ca(k,l).*Ey(i,j) + Cbx(k,l).*Hzdiffx(i,j) - Cbz(k,l).*Hxdiffz(i,j);
        
        % update to be applied only to the PML region
        PEyx(ip,j) = Bx(kp,l).*PEyx(ip,j) + Ax(kp,l).*Hzdiffx(ip,j);
        PEyx(ii,jp) = Bx(ki,lp).*PEyx(ii,jp) + Ax(ki,lp).*Hzdiffx(ii,jp);
        PEyz(ip,j) = Bz(kp,l).*PEyz(ip,j) + Az(kp,l).*Hxdiffz(ip,j);
        PEyz(ii,jp) = Bz(ki,lp).*PEyz(ii,jp) + Az(ki,lp).*Hxdiffz(ii,jp);
        Ey(ip,j) = Ey(ip,j) + Cc(kp,l).*(PEyx(ip,j) - PEyz(ip,j));
        Ey(ii,jp) = Ey(ii,jp) + Cc(ki,lp).*(PEyx(ii,jp) - PEyz(ii,jp));
        
        % add source pulse to Ey at source location
        % (emulates infinitesimal Ey directed line source with current = srcpulse)
        i = srci(s); j = srcj(s);
        Ey(i,j) = Ey(i,j) + srcpulse(it);
    
        
        % plot the Ey wavefield if necessary
        if plotopt(1)==1;   
            if mod(it-1,plotopt(2))==0
                disp(['Source ',num2str(s),'/',num2str(nsrc),', Iteration ',num2str(it),'/',num2str(numit),...
                        ':  t = ',num2str(t(it)*1e9),' ns'])
                figure(1); imagesc(xEy,zEy,Ey'); axis image
                title(['Source ',num2str(s),'/',num2str(nsrc),', Iteration ',num2str(it),'/',num2str(numit),...
                        ':  Ey wavefield at t = ',num2str(t(it)*1e9),' ns']);
                xlabel('Position (m)');  ylabel('Depth (m)');
                caxis([-plotopt(3) plotopt(3)]);
                pause(0.01);
            end
        end
    
        % record the results in gather matrix if necessary
        if mod(it-1,outstep)==0
            tout((it-1)/outstep+1) = t(it);
            for r=1:nrec
                gather((it-1)/outstep+1,r,s) = Ey(reci(r),recj(r));
            end
        end
        
    end
end
