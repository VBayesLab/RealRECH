function out = tGARCH_logPriors(theta,prior)
% LOGPRIORS calculate log of priors for tGARCH

w       = theta(:,1);
psi1    = theta(:,2);
psi2    = theta(:,3);
nu      = theta(:,4);

out  =  log(gampdf(w,prior.w_a0,1/prior.w_b0))+log(betapdf(psi2,prior.psi2_a0,prior.psi2_b0))...
    +log(gampdf(nu,prior.nu_a0,1/prior.nu_b0));
end

