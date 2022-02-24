function score = t_one_step_forecast_score(volatility_forecast,y_next,score,nu)

alpha_sig = score.alpha;

if abs(y_next)>=tinv(1-alpha_sig/2,nu)*sqrt(volatility_forecast)
    violate = 1;
else
    violate = 0;
end     
score.violate = score.violate + violate;
score.pps  = score.pps - log(pdf('tLocationScale',y_next,0,sqrt(volatility_forecast),nu));
VaR_t = tinv(alpha_sig,nu)*sqrt(volatility_forecast);
score.qs = score.qs + (y_next-VaR_t)*(alpha_sig-utils_indicator_fun(y_next,VaR_t));
score.hit = score.hit + utils_indicator_fun(y_next,VaR_t);

end

