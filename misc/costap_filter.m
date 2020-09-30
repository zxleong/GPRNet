function seis_tap = costap_filter(seis,perc)
% Function for applying a cosine taper on spesified percentage of each end 
% of the input signal.
%
% Input:
%       seis = the signal to be tapered
%       perc = the percentage of each end of the signal to be tapered
%
% Output:
%       seis_tap = the tapered signal
%
% Written by Karina Løviknes 
% 

m = length(seis); % length of the seismogram

tap = round(perc*m); % I taper off perc % of the signal 

theta = linspace(0,pi,tap);

% The first part of the costap(t) function:
costap1 = 0.5*(-cos(theta)+1);

% The middle part of the costap(t) function:
costap2 = ones(1,m-2*tap);

% The last part of the costap(t) function:
costap3 = 0.5*(cos(theta)+1);

costap = [costap1 costap2 costap3];
tm = linspace(0,m/10,m);

% figure
% plot(tm,costap)
% xlabel('Time (s)','FontSize', 14), ylabel('Costap(t)','FontSize', 14)
% title('The costap(t) function','FontSize', 18)
% axis([-20 m/10+20 0 1.1])

seis_tap = seis.*costap;

end

