function log_prior = RNN_tGARCH_logPriors(theta_particles,prior)
% LOGPRIORS calculate log of priors for RNN-GARCH

beta0   = theta_particles(:,1);
beta1   = theta_particles(:,2);
psi1    = theta_particles(:,3);
psi2    = theta_particles(:,4);
nu      = theta_particles(:,5);
w       = theta_particles(:,6);
b       = theta_particles(:,7);
v       = theta_particles(:,8:end);


log_prior  =    log(gampdf(beta0,prior.beta0_a0,1/prior.beta0_b0))...
                +log(gampdf(beta1,prior.beta1_a0,1/prior.beta1_b0))...
                +log(betapdf(psi2,prior.psi2_a0,prior.psi2_b0))...
                +log(gampdf(nu,prior.nu_a0,1/prior.nu_b0))...
                + sum(log(normpdf(v,prior.v_mu,sqrt(prior.v_var))),2)...
                + log(normpdf(w,prior.w_mu,sqrt(prior.w_var)))...
                + log(normpdf(b,prior.b_mu,sqrt(prior.b_var)));
   
end
