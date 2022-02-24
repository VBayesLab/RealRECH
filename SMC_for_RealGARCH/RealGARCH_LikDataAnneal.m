function out = RealGARCH_LikDataAnneal(data_current,data_new,mdl,mdl_in)
% Implement the SMC likelihood annealing sampler for RealGARCH
% y             : training data
% mdl           : data structure contains all neccesary settings
%
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)

y     = data_current.y;
x     = data_current.x;
y_new = data_new.y_new;
x_new = data_new.x_new;

theta_particles = mdl_in.theta_particles;
Weights         = mdl_in.Weights;
llh_calc        = mdl_in.llh_calc;  % log-likelihood of p(y_{1:t}|theta)
llh_conditional = mdl_in.llh_conditional; % log of p(y_{t+1}|y_{1:t},theta)
sigma2_cur      = mdl_in.sigma2_cur; % sigma2(t) calculated up to time t+1

% Training setting
T_anneal = 10000;                   % number of annealing steps
M        = size(theta_particles,1);% number of particles     
K        = mdl.K_lik;              % number of Markov moves
sigma20  = mdl.sigma20;            % initial volatility
prior    = mdl.prior;              % store prior setting
% Prepare for first annealing stage
psisq    = ((0:T_anneal)./T_anneal).^3;     % Design the array of annealing levels a_t
n_params = size(theta_particles,2);         % number of model parameters

t = 1;
psisq_current = psisq(t); % current annealling level a_t

while t < T_anneal+1
     t = t+1;
          
     %% Select the next annealing level a_t and then do reweighting %% 
     % Select a_t if ESS is less than the threshold. Then, reweight the particles       
     incw = log(Weights)+(psisq(t) - psisq_current).*llh_conditional;
     max_incw = max(incw);
     weights = exp(incw - max_incw);      % for numerical stabability
     Weight_trial = weights./sum(weights);     % Calculate normalized weights for current level
     ESS = 1/sum(Weight_trial.^2);             % Estimate ESS for particles in the current level
     while ESS >= 0.8*M
        t = t + 1;
        % Run until ESS at a certain level < 80%. If reach the last level,
        % the last level will be the next annealing level.
        if (t == T_anneal+1)
            incw = log(Weights)+(psisq(t)-psisq_current).*llh_conditional;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weight_trial = weights./sum(weights);
            ESS = 1/sum(Weight_trial.^2);
            break
        else % If not reach the final level -> keep checking ESS 
            incw = log(Weights)+(psisq(t)-psisq_current).*llh_conditional;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            Weight_trial = weights./sum(weights);
            ESS = 1/sum(Weight_trial.^2);
        end
     end
     Weights = Weight_trial;
     disp(['Current sub-annealing level: ',num2str(t)])     
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
     llh_calc        = llh_calc(indx);
     llh_conditional = llh_conditional(indx);
     sigma2_cur      = sigma2_cur(indx);
     Weights         = ones(M,1)/M;
                         
     %% Markov move step %%     
     log_prior    = RealGARCH_logPriors(theta_particles,prior);
     post         = log_prior+llh_calc+psisq_current*llh_conditional;   % the denominator term in the Metropolis-Hastings ratio           
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
                 log_prior_star         = RealGARCH_logPriors(theta_star,prior);             
                 [lik_star,sigma2_cur_star] = RealGARCH_llh(y,x,sigma20,theta_star);
                 [lik_conditional_star,sigma2_new_star]  = RealGARCH_llh_conditional(y_new,x_new,x(end,:),sigma2_cur_star,theta_star);                 
                 post_star      = log_prior_star + lik_star + psisq_current*lik_conditional_star; % the numerator term in the Metropolis-Hastings ratio               
                 acceptance_pro = exp(post_star-post(i));
                 acceptance_pro = min(1,acceptance_pro);                       
                 if (rand <= acceptance_pro) % if accept the new proposal sample
                     theta_particles(i,:) = theta_star;                     
                     post(i)              = post_star;
                     llh_calc(i)          = lik_star;
                     llh_conditional(i)   = lik_conditional_star;
                     sigma2_cur(i)        = sigma2_new_star;
                 end
             end             
             iter = iter + 1;
         end         
     end    
end
out.theta_particles = theta_particles;
out.Weights = Weights;
out.llh_calc = llh_calc+llh_conditional;
out.sigma2_cur = sigma2_cur;
end