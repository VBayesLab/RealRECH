function sigma2_forecast = one_step_forecast(y_cur,sigma2_cur,w,alpha,beta)

sigma2_forecast = w + beta.*y_cur^2 + alpha.*sigma2_cur;

end