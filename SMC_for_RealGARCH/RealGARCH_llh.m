function [llh,sigma2_end,llh_y] = RealGARCH_llh(y,x,sigma20,theta) 
% Calculate log-likelihood of RealGARCH with initial volatility sigma20, data y

w       = theta(:,1);
beta    = theta(:,2);
gamma   = theta(:,3);
nu      = theta(:,4);
xi      = theta(:,5);
psi     = theta(:,6);
tau1    = theta(:,7);
tau2    = theta(:,8);
sigma2u = theta(:,9);

T      = length(y);
sigma2 = zeros(T,1);

% Initialization
sigma2(1) = sigma20;
for t = 2:T
    sigma2(t) = w + beta*sigma2(t-1)+gamma*x(t-1);
end
eps_t = y./sqrt(sigma2);
u_t   = x-xi-psi*sigma2-tau1*eps_t-tau2*((nu-2)/nu*eps_t.^2-1);

llh_y = sum(log(pdf('tLocationScale',y,0,sqrt(sigma2),nu)));
llh_x = sum(-0.5*log(2*pi)-0.5*log(sigma2u)-0.5*u_t.^2/sigma2u);
llh = llh_y+llh_x;

sigma2_end = sigma2(end);

end

