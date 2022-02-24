function f = log_invgampdf(x,alpha,beta)
f = alpha.*log(beta)-gammaln(alpha)-(1+alpha).*log(x)-beta./x;

