function residual = RealGARCH_residual_analysis(y,Post_RealGARCH)

w       = mean(Post_RealGARCH.LikAnneal.w);
beta    = mean(Post_RealGARCH.LikAnneal.beta);
gamma   = mean(Post_RealGARCH.LikAnneal.gamma);
nu      = mean(Post_RealGARCH.LikAnneal.nu);
% xi      = mean(Post_RealGARCH.LikAnneal.xi);
% varpi   = mean(Post_RealGARCH.LikAnneal.varpi);
% tau1      = mean(Post_RealGARCH.LikAnneal.tau1);
% tau2      = mean(Post_RealGARCH.LikAnneal.tau2);
% sigma2u = mean(Post_RealGARCH.LikAnneal.sigma2u)

x = Post_RealGARCH.mdl.x;
T      = length(y);
sigma2 = zeros(T,1);

% Initialization
sigma2(1) = Post_RealGARCH.mdl.sigma20;
for t = 2:T
    sigma2(t) = w + beta*sigma2(t-1)+gamma*x(t-1);
end
t_residuals = y./sqrt(sigma2);
uniform_residuals = tcdf(t_residuals,nu);
std_residuals = norminv(uniform_residuals);
figure
subplot(1,3,1)
plot(std_residuals)
title('RealGARCH: Standardized residuals')
subplot(1,3,2)
[f,x] = ksdensity(std_residuals);   
plot(x,f,'-','LineWidth',2)
title('RealGARCH: Standardized residuals')
subplot(1,3,3)
qqplot(std_residuals)
title('RealGARCH: Standardized residuals QQ-plot')

residual.std_residuals = std_residuals;
residual.t_residuals = t_residuals;

end








