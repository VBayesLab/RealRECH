function Post = RealGARCH_DataAnneal(y_all,mdl)
% Implement SMC- data annealing for RealGARCH
%   y_all           : full dataset including both training and testing data          
%   mdl             : includes all necessary settings, including posterior
%                     approximation from SMC likelihhood annealing with
%                     y_train data
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


%% Training
K      = mdl.K_data;        % Number of Markov moves
prior  = mdl.prior;         % prior setting
T      = mdl.T;             % training data size
sigma20= mdl.sigma20;        % initial volatility

y       = y_all(1:T);       % training data
x_all   = mdl.x_all;
x       = x_all(1:T);         % realized volatility 
y_test  = y_all(T+1:end);
x_test  = x_all(T+1:end);% covariate time series for testing 
T_test  = length(y_test);

% Forecast score metrics
score.violate = 0;    % Number of times y true is outside forecast interval  
score.pps     = 0;    % PPS score
score.qs      = 0;    % Quantile Score
score.hit     = 0;    % Percentage of y instances below forecast VaR
score.alpha   = 0.01; % for forecast interval

% Get equally-weighted particles from SMC lik annealing as the initial particles 
theta_particles     = mdl.lik_anneal.theta_particles;
Weights             = mdl.lik_anneal.Weights;
M  = size(theta_particles,1);                    % Number of particles
n_params = size(theta_particles,2);              % Number of parameters

% Run GARCH on training data to get initialization on test data
llh_calc   = zeros(M,1);         % log-likelihood p(y_1:t|theta)
sigma2_cur = zeros(M,1);         % Store conditional variance of the current distribution
for i = 1:M
    [llh_calc(i),sigma2_cur(i)] = RealGARCH_llh(y,x,sigma20,theta_particles(i,:)); 
end

markov_idx = 0;
annealing_start = tic;
volatility_forecast = zeros(T_test,1);
for t = 0:T_test-1

    %% 1-step-ahead volatility forecast %%
    if t>0 % get current data point y_cur
        x_cur = x_test(t);  
    else
        x_cur = x(T);
    end
    sigma2_forecast = RealGARCH_one_step_forecast(x_cur,sigma2_cur,theta_particles(:,1),theta_particles(:,2),theta_particles(:,3));
    nu = theta_particles(:,4);
    nu_est = Weights'*nu;
    volatility_forecast(t+1) = nu_est/(nu_est-2)*Weights'*sigma2_forecast; % take the weighted mean as the point forecast    
    score = t_one_step_forecast_score(volatility_forecast(t+1),y_test(t+1),score,nu_est);
        
    % Calculate log conditional likelihood p(y_{t+1}|y_{1:t},theta)
    if t==0
        [llh_conditional,sigma2_cur] = RealGARCH_llh_conditional(y_test(t+1),x_test(t+1),x(T),sigma2_cur,theta_particles);
    else
        [llh_conditional,sigma2_cur] = RealGARCH_llh_conditional(y_test(t+1),x_test(t+1),x_test(t),sigma2_cur,theta_particles);
    end
    llh_calc_initial = llh_calc; % for running sub-annealing
    Weights_initial = Weights;   % for running sub-annealing 
    
    llh_calc = llh_calc+llh_conditional;
    incw = log(Weights) + llh_conditional;
    max_incw = max(incw);
    weight   = exp(incw - max_incw);    % Numerical stabability
    Weights   = weight./sum(weight);     % Calculate weights for current level
    ESS      = 1/sum(Weights.^2);             % Estimate ESS for particles in the current level           
    %% If the current particles are REALLY BAD, run likelihood annealing
    if (ESS < .4*M)
        disp(['Run sub-annealing at t = ',num2str(t)])
        mdl_in.theta_particles = theta_particles;
        mdl_in.Weights = Weights_initial;
        data_current.y = y_all(1:T+t);
        data_current.x = x_all(1:T+t);
        data_new.y_new = y_all(T+t+1);
        data_new.x_new = x_all(T+t+1);
        mdl_in.llh_calc   = llh_calc_initial;
        mdl_in.llh_conditional = llh_conditional; 
        mdl_in.sigma2_cur = sigma2_cur;
        out = RealGARCH_LikDataAnneal(data_current,data_new,mdl,mdl_in);
        theta_particles = out.theta_particles;
        Weights         = out.Weights;
        llh_calc        = out.llh_calc;
        sigma2_cur      = out.sigma2_cur;
    end        
    %% If the current particles are not good but not too bad, run Markov move. If ESS is good --> just run reweighting
    if (ESS >= .4*M)&&(ESS < .8*M)
        % calculate the covariance matrix to be used in the Random Walk MH
        % proposal. It is better to estimate this matrix BEFORE resampling
        est = sum(theta_particles.*(Weights*ones(1,n_params)));
        aux = theta_particles - ones(M,1)*est;
        V = aux'*diag(Weights)*aux;    
        C = chol(1/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale

        % Resampling for particles at the current annealing level
        indx            = utils_rs_multinomial(Weights);
        indx            = indx';
        theta_particles = theta_particles(indx,:);
        llh_calc        = llh_calc(indx);
        sigma2_cur      = sigma2_cur(indx);
        Weights = ones(M,1)./M; % reset weights after resampling 

        % Running Markov move (MH) for each paticles
        markov_idx = markov_idx + 1;
        accept = zeros(M,1);
        log_prior    = RealGARCH_logPriors(theta_particles,prior);
        post         = log_prior+llh_calc;
     
        parfor i = 1:M
            iter = 1;
            while iter<=K
                 theta = theta_particles(i,:);
                 % Using multivariate normal distribution as proposal function
                 theta_star = theta'+C*normrnd(0,1,n_params,1); theta_star = theta_star';
                 % Convert parameters to original form
                 w_star       = theta_star(:,1);
                 beta_star    = theta_star(:,2);
                 gamma_star   = theta_star(:,3);
                 nu_star      = theta_star(:,4);
                 xi_star      = theta_star(:,5);
                 psi_star     = theta_star(:,6);
                 tau1_star    = theta_star(:,7);
                 tau2_star    = theta_star(:,8);
                 sigma2u_star = theta_star(:,9);
                 if (w_star<0)||(beta_star<0)||(gamma_star<0)||(nu_star<0)||(xi_star<0)||(psi_star<0)||(sigma2u_star<0)
                     acceptance_pro = 0; % acceptance probability is zero --> do nothing
                 else                 
                     % Calculate log-posterior for proposal samples
                     log_prior_star = RealGARCH_logPriors(theta_star,prior);             
                     lik_star       = RealGARCH_llh(y_all(1:T+t+1),x_all(1:T+t+1),sigma20,theta_star);
                     post_star      = log_prior_star + lik_star;
                     acceptance_pro = exp(post_star-post(i));
                     acceptance_pro = min(1,acceptance_pro);                       
                     if (rand <= acceptance_pro) % if accept the new proposal sample
                         theta_particles(i,:) = theta_star;                     
                         post(i)              = post_star;
                         llh_calc(i)          = lik_star;
                         accept(i)            = accept(i) + 1;
                     end
                 end  
                 iter = iter + 1;
            end             
         end
         Post.accept_store(:,markov_idx) = accept/K;
         disp(['Markov move ',num2str(markov_idx),': Avarage accept rate = ',num2str(mean(accept/K))])
    end
    
end

Post.cpu = toc(annealing_start);
Post.w       = theta_particles(:,1);
Post.beta    = theta_particles(:,2);
Post.gamma   = theta_particles(:,3);
Post.nu      = theta_particles(:,4);
Post.xi      = theta_particles(:,5);
Post.psi     = theta_particles(:,6);
Post.tau1    = theta_particles(:,7);
Post.tau2    = theta_particles(:,8);
Post.sigma2u = theta_particles(:,9);
Post.theta_particles = theta_particles;
Post.M        = M;              
Post.K        = K;               
Post.W = Weights;
Post.volatility_forecast = volatility_forecast;
PPS     = score.pps/T_test;   score.PPS = PPS;
Violate = score.violate; score.Violate = Violate;
Quantile_Score = score.qs/T_test;    score.Quantile_Score = Quantile_Score;
Hit_Percentage = score.hit/T_test; score.Hit_Percentage = Hit_Percentage;
Post.score    = score;  
Model = {'RealGARCH'};
results = table(Model,PPS,Violate,Quantile_Score,Hit_Percentage);
Post.forecast = results;
disp(results);


end




