function [llh,sigma2_end] = tGARCH_llh(y,sigma20,theta_particles)
% Calculate log-likelihood of tGARCH with initial volatility sigma20, data y

w       = theta_particles(:,1);
psi1    = theta_particles(:,2);
psi2    = theta_particles(:,3);
nu      = theta_particles(:,4);

T      = length(y);
sigma2 = zeros(T,1);

% Initialization
sigma2(1) = sigma20;
for t = 2:T
    sigma2(t) = w + psi1*(1-psi2)*y(t-1)^2+ psi1*psi2*sigma2(t-1);
end

llh = sum(log(pdf('tLocationScale',y,0,sqrt(sigma2),nu)));

sigma2_end = sigma2(end);

end

