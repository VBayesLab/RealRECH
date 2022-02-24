function score = predictive_score(sigma2_proxy,sigma2_est)

score.MSE     = mean((sqrt(sigma2_est)-sqrt(sigma2_proxy)).^2);
score.MAE     = mean(abs(sqrt(sigma2_est)-sqrt(sigma2_proxy)));
score.R2LOG   = mean((log(sigma2_est./sigma2_proxy)).^2);


end

