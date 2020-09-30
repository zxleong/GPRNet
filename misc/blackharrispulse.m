function p = blackharrispulse(fr,t)
% blackharrispulse.m
%
% This function computes the derivative of a Blackman-Harris window given a time vector 
% and the desired dominant frequency.  See Chen et al. (1997), Geophysics 62, p. 1733 for 
% details.  Note that their formulation has been changed here to T = 1.14/fr, such that fr 
% represents the approximate dominant frequency of the resulting pulse.
%
% Syntax:  p = blackharrispulse(fr,t)
%
% where fr = dominant frequency (Hz)
%       t  = time vector (s)
%
% by James Irving
% July 2005

% compute the Blackman-Harris window as specified in Chen et al. (1997)
a = [0.35322222 -0.488 0.145 -0.010222222];
T = 1.14/fr;
window = zeros(size(t));
for n=0:3
    window = window + a(n+1)*cos(2*n*pi*t./T);
end
window(t>=T) = 0;

% for the pulse, approximate the window's derivative and normalize
p = window(:)';
p = [window(2:end) 0] - window(1:end);
p = p./max(abs(p));
