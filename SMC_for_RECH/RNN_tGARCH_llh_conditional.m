function [llh,sigma2_new,omega_new,h_new] = RNN_tGARCH_llh_conditional(y_new,x_new,y_cur,sigma2_cur,omega_cur,h_cur,theta_particles,act_type)
% Calculate log-likelihood of p(y_t+1 | y_{1:t},x_{t+1},theta)

beta0   = theta_particles(:,1);
beta1   = theta_particles(:,2);
psi1    = theta_particles(:,3);
psi2    = theta_particles(:,4);
nu      = theta_particles(:,5);
w       = theta_particles(:,6);
b       = theta_particles(:,7);
v       = theta_particles(:,8:end);

input = [omega_cur,y_cur*ones(length(omega_cur),1),sigma2_cur,x_new*ones(length(omega_cur),1)]; 
h_new = activation(sum(input.*v,2) + w.*h_cur + b,act_type);
omega_new    = beta0 + beta1.*h_new;
sigma2_new = omega_new + psi1.*(1-psi2).*y_cur^2 + psi1.*psi2.*sigma2_cur;

llh = log(pdf('tLocationScale',y_new,0,sqrt(sigma2_new),nu));


end