function residual = tGARCH_residual_analysis(y,Post_tGARCH)

w = mean(Post_tGARCH.LikAnneal.w);
psi1 = mean(Post_tGARCH.LikAnneal.psi1);
psi2 = mean(Post_tGARCH.LikAnneal.psi2);
nu = mean(Post_tGARCH.LikAnneal.nu);

T      = length(y);
sigma2 = zeros(T,1);
sigma2(1) = var(y);
for t = 2:T
    sigma2(t) = w + psi1*(1-psi2)*y(t-1)^2+ psi1*psi2*sigma2(t-1);
end
t_residuals = y./sqrt(sigma2);
uniform_residuals = tcdf(t_residuals,nu);
std_residuals = norminv(uniform_residuals);
figure
subplot(1,3,1)
plot(std_residuals)
title('tGARCH: Standardized residuals')
subplot(1,3,2)
[f,x] = ksdensity(std_residuals);   
plot(x,f,'-','LineWidth',2)
title('tGARCH: Standardized residuals')
subplot(1,3,3)
qqplot(std_residuals)
title('tGARCH: Standardized residuals QQ-plot')

residual.std_residuals = std_residuals;
residual.t_residuals = t_residuals;

end








