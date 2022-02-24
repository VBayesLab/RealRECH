% This example uses SMC -likelihood annealing for samppling from RNN-GARCH
% model. See the details in the math notes supplied. 
% Also, see the details and notation in Section 5 of this paper: https://arxiv.org/pdf/1908.03097.pdf

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
mdl.K1_lik   = 10;       % Number of Markov moves 
mdl.K2_lik   = 20;       % Number of Markov moves 
mdl.T        = 1500;     % size of the training time series 
y            = y_all(1:mdl.T);  % training data
mdl.x        = mdl.x_all(1:mdl.T); % covariate for training
mdl.sigma20  = var(y); %  initialize volatility in the GARCH formula with sample variance of the returns
mdl.MV_scale = 1.5;

% Prior setting: Gamma is used for the prior of beta_0 and beta_1, 
% uniform(0,1) prior is used for psi1, Beta for psi2, and normal prior for RNN
% parameters
mdl.prior.beta0_a0 = 1;     mdl.prior.beta0_b0  = 1;     % Gamma
mdl.prior.beta1_a0 = 1;     mdl.prior.beta1_b0  = 10;    % Gamma
mdl.prior.psi2_a0  = 10;    mdl.prior.psi2_b0 = 2;
mdl.prior.nu_a0    = 1;     mdl.prior.nu_b0 = 1;
mdl.prior.v_mu = 0;         mdl.prior.v_var = 0.1;    % Normal distribution
mdl.prior.w_mu = 0;         mdl.prior.w_var = 0.1;    % Normal distribution
mdl.prior.b_mu = 0;         mdl.prior.b_var = 0.1;    % Normal distribution 

mdl.act_type    = 'ReLU';        % activation function, e.g. sigmoid or ReLU 
mdl.covariate_num  = 1;     % number of covariates  

% Run Likelihood annealing for in-sample data
Post_RECH_SP500.mdl = mdl;
Post_RECH_SP500.LikAnneal = RECH_LikAnneal(y,mdl);
save('Results_RECH_SMC_SP500')


figure
subplot(2,3,1)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.beta0);   
plot(x,f,'-','LineWidth',3)
title('\beta_0')
set(gca,'FontSize',22)

subplot(2,3,2)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.beta1);   
plot(x,f,'-','LineWidth',3)
title('\beta_1')
set(gca,'FontSize',22)

subplot(2,3,3)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.alpha);   
plot(x,f,'-','LineWidth',3)
title('\alpha')
set(gca,'FontSize',22)

subplot(2,3,4)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.beta);   
plot(x,f,'-','LineWidth',3)
title('\beta')
set(gca,'FontSize',22)

subplot(2,3,5)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.b);   
plot(x,f,'-','LineWidth',3)
title('b')
set(gca,'FontSize',22)

subplot(2,3,6)
[f,x] = ksdensity(Post_RECH_SP500.LikAnneal.v(:,4));   
plot(x,f,'-','LineWidth',3)
title('v_4')
set(gca,'FontSize',22)

% Forecast with data annealing
mdl.lik_anneal          = Post_RECH_SP500.LikAnneal;
mdl.K_data              = 15;

Post_RECH_SP500.DataAnneal = RECH_DataAnneal(y_all,mdl);
save('Results_RECH_SMC_SP500')

volatility_proxy    = mdl.rv_all(mdl.T+1:end);
volatility_est      = Post_RECH_SP500.DataAnneal.volatility_forecast;
Post_RECH_SP500.DataAnneal.predictive_score = predictive_score(volatility_proxy,volatility_est);
Post_RECH_SP500.LikAnneal.residual = RECH_residual_analysis(y,Post_RECH_SP500);
save('Results_RECH_SMC_SP500')













