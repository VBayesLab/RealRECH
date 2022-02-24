function Post = tGARCH_LikAnneal(y,mdl)
% Implement the SMC likelihood annealing sampler for tGARCH
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


% Initialize particles for the first level by generating from the parameters priors
w       = gamrnd(prior.w_a0,1/prior.w_b0,M,1);
psi1    = unifrnd(0,1,M,1); 
psi2    = betarnd(prior.psi2_a0,prior.psi2_b0,M,1);
nu      = gamrnd(prior.nu_a0,1/prior.nu_b0,M,1);  % degrees of freedom
theta_particles = [w,psi1,psi2,nu]; 

% Prepare for first annealing stage
psisq    = ((0:T_anneal)./T_anneal).^3;     % Design the array of annealing levels a_t
log_llh  = 0;                               % for calculating marginal likelihood
n_params = 4;                               % number of model parameters

% Calculate log likelihood for all particles in the first annealing level
llh_calc = zeros(M,1);                      % log-likelihood calculated at each particle
for i = 1:M
    llh_calc(i) = tGARCH_llh(y,sigma20,theta_particles(i,:));            
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
     disp(['Current annealing level: ',num2str(t)])     
     psisq_current = psisq(t);
     log_llh = log_llh + log(mean(weights)) + max_incw; % log marginal likelihood
     
     % calculate the covariance matrix in the Random Walk MH proposal
     est = sum(theta_particles.*(Weights*ones(1,n_params)));
     aux = theta_particles - ones(M,1)*est;
     V = aux'*diag(Weights)*aux;    
   
     %% Resampling the particles %%
     indx     = utils_rs_multinomial(Weights');
     indx     = indx';
     theta_particles = theta_particles(indx,:);
     llh_calc = llh_calc(indx);
                  
     %% Markov move step %%     
     accept       = zeros(M,1);   % to store acceptance rate in Markov move for each particle      
     log_prior    = tGARCH_logPriors(theta_particles,prior);
     post         = log_prior+psisq_current*llh_calc;   % the denominator term in the Metropolis-Hastings ratio         
     parfor i = 1:M  % Parallelize Markov move for the particles              
         iter = 1;
         while iter<=K                                        
             theta = theta_particles(i,:); % the particle to be moved

             % Using multivariate normal distribution as proposal function
             theta_star = mvnrnd(theta,2.38/n_params*V);
             
             % Convert parameters to original form
             w_star     = theta_star(1);
             psi1_star  = theta_star(2);
             psi2_star  = theta_star(3);
             nu_star    = theta_star(4);
             if (w_star<0)||(psi1_star<0)||(psi1_star>1)||(psi2_star<0)||(psi2_star>1)||(nu_star<=0)
                 acceptance_pro = 0; % acceptance probability is zero --> do nothing
             else                 
                 % Calculate log-posterior for proposal samples
                 log_prior_star = tGARCH_logPriors(theta_star,prior);             
                 lik_star       = tGARCH_llh(y,sigma20,theta_star);                                    
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
Post.cpu      = toc(annealing_start);
Post.w        = theta_particles(:,1);
psi1          = theta_particles(:,2);
psi2          = theta_particles(:,3);
Post.nu       = theta_particles(:,4);
Post.alpha    = psi1.*(1-psi2);
Post.beta     = psi1.*psi2;
Post.psi1     = psi1;
Post.psi2     = psi2;
Post.T_anneal = T_anneal;    
Post.M        = M;              
Post.K        = K;               
Post.log_llh  = log_llh;

disp(['Marginal likelihood: ',num2str(log_llh)])


end