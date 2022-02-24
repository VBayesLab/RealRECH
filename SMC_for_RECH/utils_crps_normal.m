function f = utils_crps_normal(x,mu,sigma2)
% compute the predictive score (continuous ranked probability score - CRPS)
% for normal distribution. The smaller CRPS the better prediction. 
% See Gneiting, T., Raftery, A.: Strictly proper scoring rules, prediction, and
% estimation. J. Am. Stat. Assoc. 102, 359–378 (2007)

z = (x-mu)./sqrt(sigma2);
f = sqrt(sigma2)*(1/sqrt(pi)-2*normpdf(z)-z.*(2*normcdf(z)-1));
end
