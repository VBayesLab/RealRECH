function [llh,sigma2_end,omega_end,h_end] = RNN_tGARCH_llh(y,x,sigma20,theta_particles,act_type)
% Calculate log-likelihood of RNN-GARCH 
% INPUT
% y             : data
% x             : covariate time series
% sigma20       : initial volatility
% and the model parameters
% returns data y, covariates x and model parameters
% OUTPUT
% llh           : log-likelihood
% sigma2_end    : conditional sigma2(T) calculated at the last time T. Needed for subsequent calculation
% omega_end     : RNN component omega(T) calculated at the last time T. Needed for subsequent calculation
% h_end         : hidden layer h(T) calculated at the last time T. Needed for subsequent calculation

beta0   = theta_particles(:,1);
beta1   = theta_particles(:,2);
psi1    = theta_particles(:,3);
psi2    = theta_particles(:,4);
nu      = theta_particles(:,5);
w       = theta_particles(:,6);
b       = theta_particles(:,7);
v       = theta_particles(:,8:end);

T      = length(y);
h      = zeros(T,1);
omega    = zeros(T,1);
sigma2 = zeros(T,1);

% Initialization
h(1) = 0;
omega(1) = beta0 + beta1*h(1);
sigma2(1) = sigma20;
for t = 2:T
    input = [omega(t-1),y(t-1),sigma2(t-1),x(t,:)]'; % input to RNN
    h(t)      = activation(v*input + w*h(t-1) + b,act_type);
    omega(t)    = beta0 + beta1*h(t);
    sigma2(t) = omega(t) + psi1*(1-psi2)*y(t-1)^2 + psi1*psi2*sigma2(t-1);
end

llh = sum(log(pdf('tLocationScale',y,0,sqrt(sigma2),nu)));

sigma2_end = sigma2(end);
omega_end = omega(end);
h_end = h(end);

end
