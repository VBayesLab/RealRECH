function Post = RECH_LikAnneal(y,mdl)
% Implement the SMC likelihood annealing sampler for RECH (RNN-GARCH)
% y             : training data
% mdl           : data structure contains all neccesary settings
%
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


% Training setting
T_anneal    = mdl.T_anneal;        % number of annealing steps
M           = mdl.M;               % number of particles     
K1          = mdl.K1_lik;           % number of Markov moves in early levels
K2          = mdl.K2_lik;           % number of Markov moves in later levels, K2>=K1
sigma20     = mdl.sigma20;          % initial volatility
prior       = mdl.prior;           % store prior setting
act_type    = mdl.act_type;        % activation function, e.g. sigmoid or ReLu 
covariate_num  = mdl.covariate_num;      % number of covariates  
x           = mdl.x;               % covariate time series  
input_size  = covariate_num+3;     % number of inputs to RNN  

% Initialize particles for the first level by generating from the parameters priors
beta0 = gamrnd(prior.beta0_a0,1/prior.beta0_b0,M,1); 
beta1 = gamrnd(prior.beta1_a0,1/prior.beta1_b0,M,1); 
psi1  = unifrnd(0,1,M,1); 
psi2  = betarnd(prior.psi2_a0,prior.psi2_b0,M,1);
nu    = gamrnd(prior.nu_a0,1/prior.nu_b0,M,1);  % degrees of freedom
w     = normrnd(prior.w_mu,sqrt(prior.w_var),M,1);
b     = normrnd(prior.b_mu,sqrt(prior.b_var),M,1);
v     = normrnd(prior.v_mu,sqrt(prior.v_var),M,input_size);
theta_particles = [beta0,beta1,psi1,psi2,nu,w,b,v]; 

% Prepare for first annealing stage
psisq    = ((0:T_anneal)./T_anneal).^3;     % Design the array of annealing levels a_t
log_llh  = 0;                               % for calculating marginal likelihood
n_params = size(theta_particles,2);                               % number of model parameters

% Calculate log likelihood for all particles in the first annealing level
llh_calc = zeros(M,1);                      % log-likelihood calculated at each particle
for i = 1:M % each particle corresponds to a value of log-likelihood 
    llh_calc(i) = RNN_tGARCH_llh(y,x,sigma20,theta_particles(i,:),act_type);    
end
t = 1;
psisq_current = psisq(t); % current annealling level a_t

markov_idx = 1; % to count the times when Markov move step is executed
annealing_start = tic;
while t < T_anneal+1
     t = t+1;
          
     %% Select the next annealing level a_t and then do reweighting %% 
     % Select a_t if ESS is less than the threshold. Then, reweight the particles       
     incw = (psisq(t) - psisq_current).*llh_calc;
     max_incw = max(incw);
     weights = exp(incw - max_incw);      % for numerical stabability
     Weight = weights./sum(weights);     % Calculate normalized weights for current level
     ESS = 1/sum(Weight.^2);             % Estimate ESS for particles in the current level
     while ESS >= 0.8*M
        t = t + 1;
        % Run until ESS at a certain level < 80%. If reach the last level,
        % the last level will be the next annealing level.
        if (t == T_anneal+1)
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weight = weights./sum(weights);
            ESS = 1/sum(Weight.^2);
            break
        else % If not reach the final level -> keep checking ESS 
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weight = weights./sum(weights);
            ESS = 1/sum(Weight.^2);
        end
     end
     disp(['Current annealing level: ',num2str(t)])     
     psisq_current = psisq(t);
     log_llh = log_llh + log(mean(weights)) + max_incw; % log marginal likelihood
    
     % calculate the covariance matrix used in the Random Walk MH proposal.
     % This is the empirical covariance matrix of the particles.
     est = sum(theta_particles.*(Weight*ones(1,n_params)));
     aux = theta_particles - ones(M,1)*est;
     V = aux'*diag(Weight)*aux;           
     C = chol(mdl.MV_scale/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale

     %% Resampling the particles %%
     indx     = utils_rs_multinomial(Weight');
     indx     = indx';
     theta_particles = theta_particles(indx,:);
     llh_calc = llh_calc(indx,:); % the log-likelihood values need to match the resampled particles 
                  
     %% Markov move step %%          
     accept       = zeros(M,1);   % to store acceptance rate in Markov move for each particle      
     log_prior    = RNN_tGARCH_logPriors(theta_particles,prior);
     post         = log_prior+psisq_current*llh_calc;   % the denominator term in the Metropolis-Hastings ratio    
     if psisq_current<0.7
         K = K1;
     else
         K = K2;
     end
     parfor i = 1:M  % Parallelize Markov move for the particles              
         iter = 1;
         while iter<=K                                        
             theta = theta_particles(i,:)'; % the particle to be moved
             % Using multivariate normal distribution as proposal
             %theta_star = mvnrnd(theta,2.38/n_params*V); % 2.38/n_params is a theoretically optimal scale
             theta_star = theta+C*normrnd(0,1,n_params,1); theta_star = theta_star';
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
                 lik_star       = RNN_tGARCH_llh(y,x,sigma20,theta_star,act_type);    
                 post_star      = log_prior_star + psisq_current*lik_star; % the numerator term in the Metropolis-Hastings ratio                  
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
     Post.accept_store(:,markov_idx) = accept/K; % store acceptance rate in Markov move
     disp(['Markov move ',num2str(markov_idx),': Avarage accept rate = ',num2str(mean(accept/K))])
     markov_idx = markov_idx + 1;
     
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
Post.T_anneal = T_anneal;    
Post.M        = M;              
Post.K        = K;               
Post.log_llh  = log_llh;

disp(['Marginal likelihood: ',num2str(log_llh)])


end