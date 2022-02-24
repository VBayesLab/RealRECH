function sigma2_forecast = GARCH_one_step_forecast(y_cur,sigma2_cur,psi1,psi2,w)

sigma2_forecast = w + psi1.*(1-psi2).*y_cur^2 + psi1.*psi2.*sigma2_cur;

end