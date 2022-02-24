function [llh,sigma2_new] = tGARCH_llh_conditional(y_new,y_pre,sigma2_pre,theta_particles)
% Calculate log-likelihood of p(y_t+1 | y_1:t, theta)

w       = theta_particles(:,1);
psi1    = theta_particles(:,2);
psi2    = theta_particles(:,3);
nu      = theta_particles(:,4);

sigma2_new = w+psi1.*(1-psi2).*y_pre^2+psi1.*psi2.*sigma2_pre;

llh = log(pdf('tLocationScale',y_new,0,sqrt(sigma2_new),nu));

end

