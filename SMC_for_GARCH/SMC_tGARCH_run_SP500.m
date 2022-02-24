% This example uses SMC -likelihood annealing for samppling from tGARCH
% model with Student's error. 

clear all

%% prepare data
disp('=========== SP500 data =====================')
data       = load('../Data/realized_data.mat');
price      = data.SPX.close_price;
y_all = log(price(2:end)./price(1:end-1)); % the first day isn't included
y_all =  100*(y_all-mean(y_all)); % returns (from the 2nd day)
rv_all = 10^4*data.SPX.rv5(2:end); % realized volatility; rv_all(t) realizes the volatility of y_all(t)
y_all = y_all(end-1999:end); % use the last 2000 days
x_all = rv_all(end-2000:end-1); %IMPORTANT! By viewing realized volatility as covariate, which at any time t x_all(t) must be available,
                                % we must shift realized volatility one (1) day backward. That is, x_all(t) is
                                % realized volatility of day t-1.
mdl.rv_all = rv_all(end-1999:end);
mdl.x_all = x_all;

% Training setting
mdl.T_anneal = 10000;    % Number of pre-specified annealing steps
mdl.M        = 2000;     % Number of particles in each annealing stage
mdl.K_lik    = 20;       % Number of Markov moves 
mdl.T        = 1500;     % size of the training time series 
y            = y_all(1:mdl.T);  % training data
mdl.x        = mdl.x_all(1:mdl.T); % covariate for training
mdl.sigma20  = var(y); %  initialize volatility in the tGARCH formula with sample variance of the returns

% Prior setting: Gamma is used for the prior of w - the constant in the tGARCH formula, 
% uniform(0,1) prior is used for psi1, and Beta(10,2) is used for psi2 to encourage persistence. 
% Also, Gamma is used for the student's df
mdl.prior.w_a0 = 1; mdl.prior.w_b0 = 1;       
mdl.prior.psi2_a0 = 10; mdl.prior.psi2_b0 = 2;
mdl.prior.nu_a0 = 1; mdl.prior.nu_b0 = 1;

% Run Likelihood annealing for in-sample data
Post_tGARCH_SP500.LikAnneal = tGARCH_LikAnneal(y,mdl);
save('Results_tGARCH_SMC_SP500')

figure
subplot(2,2,1)
[f,x] = ksdensity(Post_tGARCH_SP500.LikAnneal.w);   
plot(x,f,'-','LineWidth',3)
title('\omega')
set(gca,'FontSize',22)

subplot(2,2,2)
[f,x] = ksdensity(Post_tGARCH_SP500.LikAnneal.alpha);   
plot(x,f,'-','LineWidth',3)
title('\alpha')
set(gca,'FontSize',22)

subplot(2,2,3)
[f,x] = ksdensity(Post_tGARCH_SP500.LikAnneal.beta);   
plot(x,f,'-','LineWidth',3)
title('\beta')
set(gca,'FontSize',22)

subplot(2,2,4)
[f,x] = ksdensity(Post_tGARCH_SP500.LikAnneal.nu);   
plot(x,f,'-','LineWidth',3)
title('\nu')
set(gca,'FontSize',22)


% Forecast with data annealing
mdl.lik_anneal          = Post_tGARCH_SP500.LikAnneal;
mdl.K_data              = 20;
Post_tGARCH_SP500.DataAnneal = tGARCH_DataAnneal(y_all,mdl);
save('Results_tGARCH_SMC_SP500')

volatility_proxy    = mdl.rv_all(mdl.T+1:end);
volatility_est      = Post_tGARCH_SP500.DataAnneal.volatility_forecast;
Post_tGARCH_SP500.DataAnneal.predictive_score = predictive_score(volatility_proxy,volatility_est);
Post_tGARCH_SP500.LikAnneal.residual = tGARCH_residual_analysis(y,Post_tGARCH_SP500);
save('Results_tGARCH_SMC_SP500')











