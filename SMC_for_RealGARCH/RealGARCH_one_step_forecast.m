function sigma2_forecast = RealGARCH_one_step_forecast(x_cur,sigma2_cur,w,beta,gamma)

sigma2_forecast = w + beta.*sigma2_cur+gamma.*x_cur;

end