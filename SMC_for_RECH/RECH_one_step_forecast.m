function sigma2_forecast = RECH_one_step_forecast(y_cur,x_cur,sigma2_cur,omega_cur,h_cur,theta_particles,act_type)
% each particle corresponds to a value of sigma2_forecast

beta0   = theta_particles(:,1);
beta1   = theta_particles(:,2);
psi1    = theta_particles(:,3);
psi2    = theta_particles(:,4);
nu      = theta_particles(:,5);
w       = theta_particles(:,6);
b       = theta_particles(:,7);
v       = theta_particles(:,8:end);


input = [omega_cur,y_cur*ones(length(omega_cur),1),sigma2_cur,x_cur*ones(length(omega_cur),1)]; 
h_forecast      = activation(sum(input.*v,2) + w.*h_cur + b,act_type);
omega_forecast    = beta0 + beta1.*h_forecast;
sigma2_forecast = omega_forecast + psi1.*(1-psi2).*y_cur^2 + psi1.*psi2.*sigma2_cur;

end