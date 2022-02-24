function Post = RECH_DataAnneal(y_all,mdl)
% Implement SMC- data annealing for for RECH (RNN-GARCH)
%   y_all           : full dataset including both training and testing data          
%   mdl             : includes all necessary settings, including posterior
%                     approximation from SMC likelihhood annealing with
%                     y_train data
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


%% Training
M           = mdl.M;             % Number of particles
K           = mdl.K_data;        % Number of Markov moves
prior       = mdl.prior;         % prior setting
T           = mdl.T;             % training data size
sigma20     = mdl.sigma20;       % initial volatility
act_type    = mdl.act_type;      % activation function, e.g. sigmoid or ReLu 

y           = y_all(1:T);        % training data
x           = mdl.x;             % covariate time series for training 
y_test      = y_all(T+1:end);
x_test      = mdl.x_all(T+1:end);% covariate time series for testing 
T_test      = length(y_test);

% Forecast score metrics
score.violate = 0;    % Number of times y true is outside forecast interval  
score.pps     = 0;    % PPS score
score.qs      = 0;    % Quantile Score
score.hit     = 0;    % Percentage of y instances below forecast VaR
score.alpha   = 0.01; % for forecast interval

% Get equally-weighted particles from SMC lik annealing as the initial particles 
beta0              = mdl.lik_anneal.beta0;
beta1              = mdl.lik_anneal.beta1;
psi1               = mdl.lik_anneal.psi1;
psi2               = mdl.lik_anneal.psi2;
nu                 = mdl.lik_anneal.nu;
w                  = mdl.lik_anneal.w;
b                  = mdl.lik_anneal.b;
v                  = mdl.lik_anneal.v;
theta_particles = [beta0,beta1,psi1,psi2,nu,w,b,v]; 
Weights            = ones(M,1)./M;         % Initialize equal weights for articles in the first level
n_params           = size(theta_particles,2);                               % number of model parameters

% Run GARCH on training data to get initialization on test data
llh_calc   = zeros(M,1);         % log-likelihood p(y_1:t|theta)
sigma2_cur = zeros(M,1);         % Store conditional variance of the current distribution
omega_cur  = zeros(M,1);
h_cur      = zeros(M,1);
for i = 1:M
    [llh_calc(i),sigma2_cur(i),omega_cur(i),h_cur(i)] = RNN_tGARCH_llh(y,x,sigma20,theta_particles(i,:),act_type);        
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
    sigma2_forecast = RECH_one_step_forecast(y_cur,x_test(t+1),sigma2_cur,omega_cur,h_cur,theta_particles,act_type);    
    nu = theta_particles(:,5);
    nu_est = Weights'*nu;
    volatility_forecast(t+1) = nu_est/(nu_est-2)*Weights'*sigma2_forecast; % take the weighted mean as the point forecast   
    score = t_one_step_forecast_score(volatility_forecast(t+1),y_test(t+1),score,nu_est);
    
    %% Re-weighting %%
    % Calculate log conditional likelihood p(y_t+1|y_1:t,theta)
    if t==0
        [llh_condtional,sigma2_cur,omega_cur,h_cur] = RNN_tGARCH_llh_conditional(y_test(t+1),x_test(t+1),y(T),sigma2_cur,omega_cur,h_cur,theta_particles,act_type);
    else
        [llh_condtional,sigma2_cur,omega_cur,h_cur] = RNN_tGARCH_llh_conditional(y_test(t+1),x_test(t+1),y_test(t),sigma2_cur,omega_cur,h_cur,theta_particles,act_type);
    end    
    % Update the log likelihood p(y_{1:t+1})
    llh_calc = llh_calc + llh_condtional;
    % Reweighting the particles  
    incw = log(Weights) + llh_condtional;
    max_incw = max(incw);
    weight   = exp(incw - max_incw);    % for numerical stabability
    Weights  = weight./sum(weight);     % Calculate weights for current level
    ESS      = 1/sum(Weights.^2);       % Estimate ESS for particles in the current level     
        
    %% If the current particles are not good, run resampling and Markov move
    if (ESS < 0.80*M)
        disp(['Current t: ',num2str(t)])     
        % calculate the covariance matrix to be used in the Random Walk MH
        % proposal. It is better to estimate this matrix BEFORE resampling
        est = sum(theta_particles.*(Weights*ones(1,n_params)));
        aux = theta_particles - ones(M,1)*est;
        V = aux'*diag(Weights)*aux;    
        C = chol(mdl.MV_scale/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale, used in Markov move

        % Resampling for particles at the current annealing level
        indx            = utils_rs_multinomial(Weights');
        indx            = indx';
        theta_particles = theta_particles(indx,:);
        llh_calc        = llh_calc(indx);
        sigma2_cur      = sigma2_cur(indx);
        omega_cur       = omega_cur(indx);
        h_cur           = h_cur(indx);              
        Weights = ones(M,1)./M; % reset weights after resampling 

        % Running Markov move (MH) for each paticles
        markov_idx = markov_idx + 1;
        accept = zeros(M,1);
        log_prior    = RNN_tGARCH_logPriors(theta_particles,prior);        
        post         = log_prior+llh_calc;
     
        parfor i = 1:M
            iter = 1;
            while iter<=K
                 theta = theta_particles(i,:);
                 % Using multivariate normal distribution as proposal function
                 theta_star = theta'+C*normrnd(0,1,n_params,1); theta_star = theta_star';
                 % Convert parameters to original form
                 beta0_star     = theta_star(1); 
                 beta1_star     = theta_star(2);              
                 psi1_star      = theta_star(3);
                 psi2_star      = theta_star(4);             
                 nu_star        = theta_star(5);
                 w_star         = theta_star(6);
                 b_star         = theta_star(7);
                 v_star         = theta_star(8:end);
                 if (beta0_star<0)||(beta1_star<0)||(psi1_star<0)||(psi1_star>1)||(psi2_star<0)||(psi2_star>1)||(nu_star<0)
                     acceptance_pro = 0; % acceptance probability is zero --> do nothing
                 else                 
                     % Calculate log-posterior for proposal samples
                     log_prior_star = RNN_tGARCH_logPriors(theta_star,prior);
                     [lik_star,sigma2_new_star,omega_new_star,h_new_star] = RNN_tGARCH_llh(y_all(1:T+t+1),mdl.x_all(1:T+t+1),sigma20,theta_star,act_type);                         
                     post_star      = log_prior_star + lik_star; % the numerator term in the Metropolis-Hastings ratio 
                     acceptance_pro = exp(post_star-post(i));
                     acceptance_pro = min(1,acceptance_pro);                       
                     if (rand <= acceptance_pro) % if accept the new proposal sample
                         theta_particles(i,:) = theta_star;                     
                         post(i)              = post_star;
                         llh_calc(i)          = lik_star;
                         sigma2_cur(i)        = sigma2_new_star;
                         omega_cur(i)         = omega_new_star;
                         h_cur(i)             = h_new_star;
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
Post.cpu     = toc(annealing_start);
Post.beta0   = theta_particles(:,1);
Post.beta1   = theta_particles(:,2);
Post.psi1    = theta_particles(:,3);
Post.psi2    = theta_particles(:,4);
Post.nu      = theta_particles(:,5);
Post.w       = theta_particles(:,6);
Post.b       = theta_particles(:,7);
Post.v       = theta_particles(:,8:end);
Post.alpha   = Post.psi1.*(1-Post.psi2);
Post.beta    = Post.psi1.*Post.psi2;
Post.M        = M;              
Post.K        = K;               
Post.Weights  = Weights;
Post.volatility_forecast = volatility_forecast;
PPS     = score.pps/T_test;   score.PPS = PPS;
Violate = score.violate; score.Violate = Violate;
Quantile_Score = score.qs/T_test;    score.Quantile_Score = Quantile_Score;
Hit_Percentage = score.hit/T_test; score.Hit_Percentage = Hit_Percentage;
Post.score    = score;  
Model = {'RNN-GARCH'};
results = table(Model,PPS,Violate,Quantile_Score,Hit_Percentage);
Post.forecast = results;
disp(results);

end







