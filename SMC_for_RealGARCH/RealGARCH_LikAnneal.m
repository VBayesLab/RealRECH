function Post = RealGARCH_LikAnneal(y,mdl)
% Implement the SMC likelihood annealing sampler for RealGARCH
% y             : training data
% mdl           : data structure contains all neccesary settings
%
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


% Training setting
T_anneal = mdl.T_anneal;        % number of annealing steps
M        = mdl.M;               % number of particles     
K        = mdl.K_lik;           % number of Markov moves
sigma20  = mdl.sigma20;         % initial volatility
prior    = mdl.prior;           % store prior setting
x        = mdl.x;               % realized volatilities   

% Initialize particles for the first level by generating from the parameters priors
w       = gamrnd(prior.w_a0,1/prior.w_b0,M,1);
beta    = betarnd(prior.beta_a0,prior.beta_b0,M,1);
gamma   = betarnd(prior.gamma_a0,1/prior.gamma_b0,M,1);
nu      = gamrnd(prior.nu_a0,1/prior.nu_b0,M,1);  % degrees of freedom
xi      = gamrnd(prior.xi_a0,1/prior.xi_b0,M,1);
psi     = gamrnd(prior.psi_a0,1/prior.psi_b0,M,1);
tau1    = normrnd(prior.tau1_mu,sqrt(prior.tau1_var),M,1);
tau2    = normrnd(prior.tau2_mu,sqrt(prior.tau2_var),M,1);
sigma2u = gamrnd(prior.sigma2u_a0,1/prior.sigma2u_b0,M,1);
theta_particles = [w,beta,gamma,nu,xi,psi,tau1,tau2,sigma2u]; 
Weights = ones(M,1)/M;

% Prepare for first annealing stage
psisq    = ((0:T_anneal)./T_anneal).^3;     % Design the array of annealing levels a_t
log_llh  = 0;                               % for calculating marginal likelihood log p(y)
n_params = 9;                               % number of model parameters

% Calculate log likelihood for all particles in the first annealing level
llh_calc = zeros(M,1);                      % log-likelihood calculated at each particle
llh_calc_y = zeros(M,1);                    % log-likelihood (for y) calculated at each particle
for i = 1:M
    [llh_calc(i),~,llh_calc_y(i)] = RealGARCH_llh(y,x,sigma20,theta_particles(i,:)); 
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
     Weights = weights./sum(weights);     % Calculate normalized weights for current level
     ESS = 1/sum(Weights.^2);             % Estimate ESS for particles in the current level
     while ESS >= 0.8*M
        t = t + 1;
        % Run until ESS at a certain level < 80%. If reach the last level,
        % the last level will be the next annealing level.
        if (t == T_anneal+1)
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weights = weights./sum(weights);
            ESS = 1/sum(Weights.^2);
            break
        else % If not reach the final level -> keep checking ESS 
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weights = weights./sum(weights);
            ESS = 1/sum(Weights.^2);
        end
     end     
     % for calculating marginal likelihood for y
     lw = (psisq(t) - psisq_current).*llh_calc_y;
     max_lw = max(lw);
     weights = exp(lw - max_lw);      % for numerical stabability
     log_llh = log_llh + log(mean(weights)) + max_lw; % log marginal likelihood
     
     disp(['Current annealing level: ',num2str(t)])     
     psisq_current = psisq(t);
   
     % calculate the covariance matrix in the Random Walk MH proposal
     est = sum(theta_particles.*(Weights*ones(1,n_params)));
     aux = theta_particles - ones(M,1)*est;
     V = aux'*diag(Weights)*aux;  
     C = chol(1.5/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale
    
     %% Resampling the particles %%
     indx     = utils_rs_multinomial(Weights');
     indx     = indx';
     theta_particles = theta_particles(indx,:);
     Weights         = ones(M,1)/M;
     llh_calc        = llh_calc(indx);
     llh_calc_y      = llh_calc_y(indx);
                  
     %% Markov move step %%     
     accept       = zeros(M,1);   % to store acceptance rate in Markov move for each particle      
     log_prior    = RealGARCH_logPriors(theta_particles,prior);
     post         = log_prior+psisq_current*llh_calc;   % the denominator term in the Metropolis-Hastings ratio      
     
     parfor i = 1:M  % Parallelize Markov move for the particles              
         iter = 1;
         while iter<=K                                        
             theta = theta_particles(i,:)'; % the particle to be moved
             % Using multivariate normal distribution as proposal function
             theta_star = theta+C*normrnd(0,1,n_params,1); theta_star = theta_star';
             
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
                 [lik_star,~,lik_y_star]       = RealGARCH_llh(y,x,sigma20,theta_star);
                 post_star      = log_prior_star + psisq_current*lik_star; % the numerator term in the Metropolis-Hastings ratio               
                 acceptance_pro = exp(post_star-post(i));
                 acceptance_pro = min(1,acceptance_pro);                       
                 if (rand <= acceptance_pro) % if accept the new proposal sample
                     theta_particles(i,:) = theta_star;                     
                     post(i)              = post_star;
                     llh_calc(i)          = lik_star;
                     llh_calc_y(i)        = lik_y_star;
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
Post.Weights = Weights;
Post.T_anneal = T_anneal;    
Post.M        = M;              
Post.K        = K;               
Post.log_llh  = log_llh;

disp(['Marginal likelihood: ',num2str(log_llh)])


end