function residual = RECH_residual_analysis(y,Post_RECH)

beta0 = mean(Post_RECH.LikAnneal.beta0);
beta1 = mean(Post_RECH.LikAnneal.beta1);
psi1 = mean(Post_RECH.LikAnneal.psi1);
psi2 = mean(Post_RECH.LikAnneal.psi2);
nu = mean(Post_RECH.LikAnneal.nu);
v = mean(Post_RECH.LikAnneal.v);
w = mean(Post_RECH.LikAnneal.w);
b = mean(Post_RECH.LikAnneal.b);

x = Post_RECH.mdl.x;
T      = length(y);
h      = zeros(T,1);
omega    = zeros(T,1);
sigma2 = zeros(T,1);
sigma20 = var(y);

h(1) = 0;
omega(1) = beta0 + beta1*h(1);
sigma2(1) = sigma20;
for t = 2:T
    input = [omega(t-1),y(t-1),sigma2(t-1),x(t)]'; % input to RNN
    h(t)      = activation(v*input + w*h(t-1) + b,Post_RECH.mdl.act_type);
    omega(t)    = beta0 + beta1*h(t);
    sigma2(t) = omega(t) + psi1*(1-psi2)*y(t-1)^2 + psi1*psi2*sigma2(t-1);
end

t_residuals = y./sqrt(sigma2);
uniform_residuals = tcdf(t_residuals,nu);
std_residuals = norminv(uniform_residuals);
figure
subplot(1,3,1)
plot(std_residuals)
title('RECH: Standardized residuals')
subplot(1,3,2)
[f,x] = ksdensity(std_residuals);   
plot(x,f,'-','LineWidth',2)
title('RECH: Standardized residuals')
subplot(1,3,3)
qqplot(std_residuals)
title('RECH: Standardized residuals QQ-plot')

residual.std_residuals = std_residuals;
residual.t_residuals = t_residuals;

end








