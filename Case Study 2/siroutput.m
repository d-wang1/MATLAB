%% This function takes three inputs
% x - a set of parameters
% t - the number of time-steps you wish to simulate
% data - actual data that you are attempting to fit

function f = siroutput(x,t,data)

% set up transmission constants
k_infections = x(1);
k_fatality = x(2);
k_recover = x(3);

% set up initial conditions
ic_susc = x(4);
ic_inf = x(5);
ic_rec = x(6);
ic_fatality = x(7);

% Set up SIRD within-population transmission matrix
% A = [1-k_infections k_infections 0 0,0 1-k_recover-k_fatality k_recover k_fatality, 0 0 1 0, 0 0 0 1];
A = [1-k_infections 0 0 0; k_infections 1-k_recover-k_fatality 0 0; 0 k_recover 1 0; 0 k_fatality 0 1];
B = zeros(4,1);

% Set up the vector of initial conditions
x0 = [ic_susc, ic_inf, ic_rec, ic_fatality];

% simulate the SIRD model for t time-steps
sys_sir_base = ss(A,B,eye(4),zeros(4,1),1)
y = lsim(sys_sir_base,zeros(t,1),linspace(0,t-1,t),x0);
disp(y)
% return a "cost".  This is the quantitity that you want your model to
% minimize.  Basically, this should encapsulate the difference between your
% modeled data and the true data. Norms and distances will be useful here.
% Hint: This is a central part of this case study!  choices here will have
% a big impact!
f = norm(abs(data(:,1))-abs(y(:,2)))+norm(abs(data(:,2))-abs(y(:,4)));

end