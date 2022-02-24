% This example uses SMC -likelihood annealing for samppling from tGARCH
% model with Student's error. 

clear all

%% prepare data
disp('=========== SP500 data =====================')
data   = load('../Data/realized_data.mat');
price  = data.SPX.close_price;
y_all  = log(price(2:end)./price(1:end-1)); % the first day isn't included
y_all  = 100*(y_all-mean(y_all)); % returns (from the 2nd day)
rv_all = 10^4*data.SPX.rv5(2:end); % realized volatility; rv_all(t) realizes the volatility of y_all(t)

y_all  = y_all(end-1999:end); % use the last 2000 days
mdl.x_all = rv_all(end-1999:end);

% Training setting
mdl.T_anneal = 10000;    % Number of pre-specified annealing steps
mdl.M        = 2000;     % Number of particles in each annealing stage
mdl.K_lik    = 20;       % Number of Markov moves 
mdl.T        = 1500;     % size of the training time series 
y            = y_all(1:mdl.T);  % training data
mdl.x        = mdl.x_all(1:mdl.T); % rv for training
mdl.sigma20  = var(y); %  initialize volatility in the tGARCH formula with sample variance of the returns

% Prior setting: 
mdl.prior.w_a0 = 1;         mdl.prior.w_b0 = 1;         %Gamma
mdl.prior.beta_a0 = 10;     mdl.prior.beta_b0 = 2;      %Beta
mdl.prior.gamma_a0 = 2;     mdl.prior.gamma_b0 = 5;     %Beta
mdl.prior.nu_a0 = 1;        mdl.prior.nu_b0 = 1;        %Gamma
mdl.prior.xi_a0 = 1;        mdl.prior.xi_b0 = 1;        %Gamma
mdl.prior.psi_a0 = 1;       mdl.prior.psi_b0 = 1;       %Gamma
mdl.prior.tau1_mu = 0;      mdl.prior.tau1_var = .1;      %Normal
mdl.prior.tau2_mu = 0;      mdl.prior.tau2_var = .1;      %Normal
mdl.prior.sigma2u_a0 = 1;   mdl.prior.sigma2u_b0 = 5; %Gamma

% Run Likelihood annealing for in-sample data
Post_RealGARCH_SP500.mdl = mdl;
Post_RealGARCH_SP500.LikAnneal = RealGARCH_LikAnneal(y,mdl);
Post_RealGARCH_SP500.LikAnneal.residual = RealGARCH_residual_analysis(y,Post_RealGARCH_SP500);
save('Results_RealGARCH_SMC_SP500')


figure
subplot(2,2,1)
[f,x] = ksdensity(Post_RealGARCH_SP500.LikAnneal.w);   
plot(x,f,'-','LineWidth',3)
title('\omega')
set(gca,'FontSize',22)

subplot(2,2,2)
[f,x] = ksdensity(Post_RealGARCH_SP500.LikAnneal.beta);   
plot(x,f,'-','LineWidth',3)
title('\beta')
set(gca,'FontSize',22)

subplot(2,2,3)
[f,x] = ksdensity(Post_RealGARCH_SP500.LikAnneal.nu);   
plot(x,f,'-','LineWidth',3)
title('\nu')
set(gca,'FontSize',22)

subplot(2,2,4)
[f,x] = ksdensity(Post_RealGARCH_SP500.LikAnneal.psi);   
plot(x,f,'-','LineWidth',3)
title('\psi')
set(gca,'FontSize',22)


% Forecast with data annealing
mdl.lik_anneal          = Post_RealGARCH_SP500.LikAnneal;
mdl.K_data              = 20;
Post_RealGARCH_SP500.DataAnneal = RealGARCH_DataAnneal(y_all,mdl);
save('Results_RealGARCH_SMC_SP500')

volatility_proxy    = mdl.x_all(mdl.T+1:end);
volatility_est      = Post_RealGARCH_SP500.DataAnneal.volatility_forecast;
Post_RealGARCH_SP500.DataAnneal.predictive_score = predictive_score(volatility_proxy,volatility_est);
save('Results_RealGARCH_SMC_SP500')











