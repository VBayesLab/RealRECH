function Post = tGARCH_DataAnneal(y_all,mdl)
% Implement SMC- data annealing for tGARCH
%   y_all           : full dataset including both training and testing data          
%   mdl             : includes all necessary settings, including posterior
%                     approximation from SMC likelihhood annealing with
%                     y_train data
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


%% Training
M      = mdl.M;             % Number of particles
K      = mdl.K_data;        % Number of Markov moves
prior  = mdl.prior;         % prior setting
T      = mdl.T;             % training data size
sigma20 = mdl.sigma20;        % initial volatility

y       = y_all(1:T);       % training data
y_test  = y_all(T+1:end);
T_test  = length(y_test);

% Forecast score metrics
score.violate = 0;    % Number of times y true is outside forecast interval  
score.pps     = 0;    % PPS score
score.qs      = 0;    % Quantile Score
score.hit     = 0;    % Percentage of y instances below forecast VaR
score.alpha   = 0.01; % for forecast interval

% Get equally-weighted particles from SMC lik annealing as the initial particles 
psi1                = mdl.lik_anneal.psi1;
psi2                = mdl.lik_anneal.psi2;
w                   = mdl.lik_anneal.w;
nu                  = mdl.lik_anneal.nu;
theta_particles     = [w,psi1,psi2,nu];
Weights             = ones(M,1)./M;         % Initialize equal weights for articles in the first level
n_params = 4;                    % Number of parameters

% Run GARCH on training data to get initialization on test data
llh_calc   = zeros(M,1);         % log-likelihood p(y_1:t|theta)
sigma2_cur = zeros(M,1);         % Store conditional variance of the current distribution
for i = 1:M
    [llh_calc(i),sigma2_cur(i)] = tGARCH_llh(y,sigma20,theta_particles(i,:));
end

markov_idx = 0;
annealing_start = tic;
volatility_forecast = zeros(T_test,1);
for t = 0:T_test-1

    %% 1-step-ahead volatility forecast %%
    if t>0 % get current data point y_cur
        y_cur = y_test(t);  
    else
        y_cur = y(T);
    end
    sigma2_forecast = GARCH_one_step_forecast(y_cur,sigma2_cur,theta_particles(:,2),theta_particles(:,3),theta_particles(:,1));    
    nu = theta_particles(:,4);
    nu_est = Weights'*nu;
    volatility_forecast(t+1) = nu_est/(nu_est-2)*Weights'*sigma2_forecast; % take the weighted mean as the point forecast    
    score = t_one_step_forecast_score(volatility_forecast(t+1),y_test(t+1),score,nu_est);
    
    %% Re-weighting %%
    % Calculate log conditional likelihood p(y_t+1|y_1:t,theta)
    if t==0
        [llh_condtional,sigma2_cur] = tGARCH_llh_conditional(y_test(t+1),y(T),sigma2_cur,theta_particles);
    else
        [llh_condtional,sigma2_cur] = tGARCH_llh_conditional(y_test(t+1),y_test(t),sigma2_cur,theta_particles);
    end    
    % Update the log likelihood p(y_{1:t+1})
    llh_calc = llh_calc + llh_condtional;
    % Reweighting the particles  
    incw = log(Weights) + llh_condtional;
    max_incw = max(incw);
    weight   = exp(incw - max_incw);    % Numerical stabability
    Weights   = weight./sum(weight);     % Calculate weights for current level
    ESS      = 1/sum(Weights.^2);             % Estimate ESS for particles in the current level   
        
    %% If the current particles are not good, run resampling and Markov move
    if (ESS < 0.80*M)
        % calculate the covariance matrix to be used in the Random Walk MH
        % proposal. It is better to estimate this matrix BEFORE resampling
        est = sum(theta_particles.*(Weights*ones(1,n_params)));
        aux = theta_particles - ones(M,1)*est;
        V = aux'*diag(Weights)*aux;    

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
        log_prior    = tGARCH_logPriors(theta_particles,prior);
        post         = log_prior+llh_calc;
     
        parfor i = 1:M
            iter = 1;
            while iter<=K
                 theta = theta_particles(i,:);
                 % Using multivariate normal distribution as proposal function
                 theta_star = mvnrnd(theta,2.38/n_params*V);
                 w_star     = theta_star(1);
                 psi1_star  = theta_star(2);
                 psi2_star  = theta_star(3);
                 nu_star    = theta_star(4);
                 if (w_star<0)||(psi1_star<0)||(psi1_star>1)||(psi2_star<0)||(psi2_star>1)||(nu_star<=0)
                     acceptance_pro = 0; % acceptance probability is zero --> do nothing
                 else                 
                     % Calculate log-posterior for proposal samples
                     log_prior_star = tGARCH_logPriors(theta_star,prior);             
                     lik_star       = tGARCH_llh(y_all(1:T+t+1),sigma20,theta_star);  % log-likelihood of y_{1:T+t+1}                     
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
         disp(['Markov move at t = ',num2str(t),': Avarage accept rate = ',num2str(mean(accept/K))])
    end
    
end
Post.cpu = toc(annealing_start);
Post.w        = theta_particles(:,1);
psi1          = theta_particles(:,2);
psi2          = theta_particles(:,3);
Post.nu       = theta_particles(:,4);
Post.alpha    = psi1.*(1-psi2);
Post.beta     = psi1.*psi2;
Post.psi1     = psi1;
Post.psi2     = psi2;
Post.M        = M;              
Post.K        = K;               
Post.Weights  = Weights;
Post.volatility_forecast = volatility_forecast;
PPS     = score.pps/T_test;   score.PPS = PPS;
Violate = score.violate; score.Violate = Violate;
Quantile_Score = score.qs/T_test;    score.Quantile_Score = Quantile_Score;
Hit_Percentage = score.hit/T_test; score.Hit_Percentage = Hit_Percentage;
Post.score    = score;  
Model = {'tGARCH'};
results = table(Model,PPS,Violate,Quantile_Score,Hit_Percentage);
Post.forecast = results;
disp(results);

end







