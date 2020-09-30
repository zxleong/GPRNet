function [rw,t] = ricker(f,n,dt,t0,t1)
%RICKER creates an causal ricker wavelet signal
%
%   RICKER creates and plots a default causal ricker wavelet with:
%
%       peak frequency   = 20 Hz
%       sampling time    = 0.001 seconds
%       number of points = 100;
%       peak location    = 1/F = 1/20Hz
%
%   RW = RICKER(...) returns the default wavelet in RW.
%
%   [RW,T] = RICKER(...) returns the time vector in T.
%
%   Specifying parameters of peak frequency (F, Hz), number of points (N),
%   and sampling time (DT) are specified by the syntax:
%
%       [RW,T] = RICKER(F)
%       [RW,T] = RICKER(F,N)
%       [RW,T] = RICKER(F,N,DT)
%       
%   [RW,T] = RICKER(F,N,DT,T0) creates a ricker wavelet with peak centered
%   at T0.
%
%   [RW,T] = RICKER(F,N,DT,T0,T1) creates a 2 dimensional symmetric
%   ricker wavelet with sift in 1st dimension of T0 and second dimension of
%   T1.
%
%   Example 1:
%       ricker % plots a 20 Hz Ricker Wavelet over 0.1 seconds
%
%   Example 2:
%    % create a ricker wavelet with 40 Hz, 200 points, and 0.02 s between
%    % samples
%    [rw,t] = ricker(40,200,0.002);
%    plot(t,rw), xlabel('Time'), ylabel('Amplitude')

% Define inputs if needed
switch nargin
    case 0
        f  = 20;
        n  = 100;
        dt = 0.001;
        t0 = 1/f;
        is2d = false;
    case 1
        n = 100;
        dt = 0.001;
        t0 = 1/f;
        is2d = false;
    case 2
        dt = 0.001;
        t0 = 1/f;
        is2d = false;
    case 3
        t0 = 1/f;
        is2d = false;
    case 4 % use all values
        is2d = false;
    case 5 % use all inputs
        is2d = true;
    otherwise
        warning('RICKER:tooManyInputs','Ignoring additional inputs')
end

% Create the wavelet and shift in time if needed
T = dt*(n-1);
t = 0:dt:T;
tau = t-t0;
if ~is2d
    s = (1-tau.*tau*f^2*pi^2).*exp(-tau.^2*pi^2*f^2);
else
    [t1,t2] = meshgrid(tau,t-t1);
    s = (1-(t1.^2+t2.^2)*f^2*pi^2).*exp(-(t1.^2+t2.^2)*pi^2*f^2);
end

if nargout == 0
        plot(t,s)
        xlabel('Time (s)')
        ylabel('Amplitude')
        title('Ricker Wavelet')
else
        rw = s;
end