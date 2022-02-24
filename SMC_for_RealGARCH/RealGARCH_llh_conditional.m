function [llh,sigma2_new,llh_y] = RealGARCH_llh_conditional(y_new,x_new,x_cur,sigma2_cur,theta)
% Calculate log-likelihood of p(y_{t+1},x_{t+1} | y_1:t,x_1:t, theta)

w       = theta(:,1);
beta    = theta(:,2);
gamma   = theta(:,3);
nu      = theta(:,4);
xi      = theta(:,5);
psi     = theta(:,6);
tau1    = theta(:,7);
tau2    = theta(:,8);
sigma2u = theta(:,9);

sigma2_new = w+beta.*sigma2_cur+gamma.*x_cur;
eps_new = y_new./sqrt(sigma2_new);
u_new = x_new - xi - psi.*sigma2_new-tau1.*eps_new-tau2.*((nu-2)./nu.*eps_new.^2-1);

llh_y = log(pdf('tLocationScale',y_new,0,sqrt(sigma2_new),nu));
llh_x = -0.5*log(2*pi)-0.5*log(sigma2u)-0.5*u_new.^2./sigma2u;
llh = llh_y+llh_x;

end

