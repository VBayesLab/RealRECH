% This scipt is to reproduce the SP500 results reported in the paper
% "Realized Volatility Recurrent Conditional Heteroscedasticity" by TRan et al. 

clc
clear all
disp('=====================================================================')
disp('=========================  SP500  ===================================')
disp('=====================================================================')

disp('====================== GARCH: SP500 data =============================')
load('SMC_for_GARCH/Results_tGARCH_SMC_SP500')

mean_w = mean(Post_tGARCH_SP500.LikAnneal.w); std_w = std(Post_tGARCH_SP500.LikAnneal.w);
w = [mean_w;std_w];
mean_alpha = mean(Post_tGARCH_SP500.LikAnneal.alpha); std_alpha = std(Post_tGARCH_SP500.LikAnneal.alpha);
alpha = [mean_alpha;std_alpha];
mean_beta = mean(Post_tGARCH_SP500.LikAnneal.beta);std_beta = std(Post_tGARCH_SP500.LikAnneal.beta);
beta = [mean_beta;std_beta];
mean_nu = mean(Post_tGARCH_SP500.LikAnneal.nu);std_nu = std(Post_tGARCH_SP500.LikAnneal.nu);
nu = [mean_nu;std_nu];
llh = [Post_tGARCH_SP500.LikAnneal.log_llh;0];
posterior = {'mean','std'};
garch_table = table(posterior',w,alpha,beta,nu,llh);
garch_table.Properties.VariableNames = {'post','w', 'alpha','beta','nu','llh'};
disp(garch_table)

disp('===================== RealGARCH: SP500 data ==========================')
clear all
load('SMC_for_RealGARCH/Results_RealGARCH_SMC_SP500')
mean_gamma = mean(Post_RealGARCH_SP500.LikAnneal.gamma);std_gamma = std(Post_RealGARCH_SP500.LikAnneal.gamma);
gamma = [mean_gamma;std_gamma];
mean_beta = mean(Post_RealGARCH_SP500.LikAnneal.beta);std_beta = std(Post_RealGARCH_SP500.LikAnneal.beta);
beta = [mean_beta;std_beta];
mean_nu = mean(Post_RealGARCH_SP500.LikAnneal.nu);std_nu = std(Post_RealGARCH_SP500.LikAnneal.nu);
nu = [mean_nu;std_nu];
mean_psi = mean(Post_RealGARCH_SP500.LikAnneal.psi);std_psi = std(Post_RealGARCH_SP500.LikAnneal.psi);
psi = [mean_psi;std_psi];
llh = [Post_RealGARCH_SP500.LikAnneal.log_llh;0];
posterior = {'mean','std'};
realgarch_table = table(posterior',gamma,beta,nu,psi,llh);
realgarch_table.Properties.VariableNames = {'post','gamma', 'beta','nu','varphi','llh'};
disp(realgarch_table)

disp('===================== RECH: SP500 data ================================')
clear all
load('SMC_for_RECH/Results_RECH_SMC_SP500')
mean_alpha = mean(Post_RECH_SP500.LikAnneal.alpha);std_alpha = std(Post_RECH_SP500.LikAnneal.alpha);
alpha = [mean_alpha;std_alpha];
mean_beta = mean(Post_RECH_SP500.LikAnneal.beta);std_beta = std(Post_RECH_SP500.LikAnneal.beta);
beta = [mean_beta;std_beta];
mean_nu = mean(Post_RECH_SP500.LikAnneal.nu);std_nu = std(Post_RECH_SP500.LikAnneal.nu);
nu = [mean_nu;std_nu];
mean_beta1 = mean(Post_RECH_SP500.LikAnneal.beta1);std_beta1 = std(Post_RECH_SP500.LikAnneal.beta1);
beta1 = [mean_beta1;std_beta1];
v_rv = Post_RECH_SP500.LikAnneal.v(:,4);
mean_v_rv = mean(v_rv);std_v_rv = std(v_rv);
v_rv = [mean_v_rv;std_v_rv];
llh = [Post_RECH_SP500.LikAnneal.log_llh;0];
posterior = {'mean','std'};
rech_table = table(posterior',alpha,beta,nu,beta1,v_rv,llh);
rech_table.Properties.VariableNames = {'post','alpha', 'beta','nu','beta1','v_rv','llh'};
disp(rech_table)


load('SMC_for_GARCH/Results_tGARCH_SMC_SP500')
garch_res = Post_tGARCH_SP500.LikAnneal.residual.std_residuals;
garch_tres = Post_tGARCH_SP500.LikAnneal.residual.t_residuals;
load('SMC_for_RealGARCH/Results_RealGARCH_SMC_SP500')
realgarch_res = Post_RealGARCH_SP500.LikAnneal.residual.std_residuals;
realgarch_tres = Post_RealGARCH_SP500.LikAnneal.residual.t_residuals;
load('SMC_for_RECH/Results_RECH_SMC_SP500')
rech_res = Post_RECH_SP500.LikAnneal.residual.std_residuals;
rech_tres = Post_RECH_SP500.LikAnneal.residual.t_residuals;


%======================= Plot of forecast interval ==============================
y_test = y_all(mdl.T+1:end);
l = length(y_test);
plot(y_test)
hold on
garch_vl = sqrt(Post_tGARCH_SP500.DataAnneal.volatility_forecast);
realgarch_vl = sqrt(Post_RealGARCH_SP500.DataAnneal.volatility_forecast);
rech_vl = sqrt(Post_RECH_SP500.DataAnneal.volatility_forecast);
plot(1:l,-2*garch_vl,'--r',1:l,2*garch_vl,'--r');
plot(1:l,-2*realgarch_vl,'-.b',1:l,2*realgarch_vl,'-.b');
plot(1:l,-2*rech_vl,'-k',1:l,2*rech_vl,'-k');
hold off





disp('======================= Residual table ==============================')
mean_tres = [mean(garch_tres);mean(realgarch_tres);mean(rech_tres)];
std_tres = [std(garch_tres);std(realgarch_tres);std(rech_tres)];
skew_tres = [skewness(garch_tres);skewness(realgarch_tres);skewness(rech_tres)];
kurtosis_tres = [kurtosis(garch_tres);kurtosis(realgarch_tres);kurtosis(rech_tres)];
model = {'GARCH','RealGARCH','RECH'};
tres_table = table(model',mean_tres,std_tres,skew_tres,kurtosis_tres);
tres_table.Properties.VariableNames = {'Model: t-residual','mean', 'std','skewness','kurtosis'};
disp(tres_table)

disp('================== Normalized residual table ========================')
mean_res = [mean(garch_res);mean(realgarch_res);mean(rech_res)];
std_res = [std(garch_res);std(realgarch_res);std(rech_res)];
skew_res = [skewness(garch_res);skewness(realgarch_res);skewness(rech_res)];
kurtosis_res = [kurtosis(garch_res);kurtosis(realgarch_res);kurtosis(rech_res)];
model = {'GARCH','RealGARCH','RECH'};
res_table = table(model',mean_res,std_res,skew_res,kurtosis_res);
res_table.Properties.VariableNames = {'Model: normalized residual','mean', 'std','skewness','kurtosis'};
disp(res_table)


figure
subplot(3,2,1)
plot(garch_res)
subplot(3,2,2)
qqplot(garch_res)
subplot(3,2,3)
plot(realgarch_res)
subplot(3,2,4)
qqplot(realgarch_res)
subplot(3,2,5)
plot(rech_res)
subplot(3,2,6)
qqplot(rech_res)

disp('================== Prediction Performance  =========================')
PPS = [Post_tGARCH_SP500.DataAnneal.score.PPS;Post_RealGARCH_SP500.DataAnneal.score.PPS;Post_RECH_SP500.DataAnneal.score.PPS];
Violate = [Post_tGARCH_SP500.DataAnneal.score.Violate;Post_RealGARCH_SP500.DataAnneal.score.Violate;Post_RECH_SP500.DataAnneal.score.Violate];
Quantile_score = [Post_tGARCH_SP500.DataAnneal.score.Quantile_Score;Post_RealGARCH_SP500.DataAnneal.score.Quantile_Score;Post_RECH_SP500.DataAnneal.score.Quantile_Score];
Hit = [Post_tGARCH_SP500.DataAnneal.score.Hit_Percentage;Post_RealGARCH_SP500.DataAnneal.score.Hit_Percentage;Post_RECH_SP500.DataAnneal.score.Hit_Percentage];
MSE = [Post_tGARCH_SP500.DataAnneal.predictive_score.MSE;Post_RealGARCH_SP500.DataAnneal.predictive_score.MSE;Post_RECH_SP500.DataAnneal.predictive_score.MSE];
MAE = [Post_tGARCH_SP500.DataAnneal.predictive_score.MAE;Post_RealGARCH_SP500.DataAnneal.predictive_score.MAE;Post_RECH_SP500.DataAnneal.predictive_score.MAE];
R2LOG = [Post_tGARCH_SP500.DataAnneal.predictive_score.R2LOG;Post_RealGARCH_SP500.DataAnneal.predictive_score.R2LOG;Post_RECH_SP500.DataAnneal.predictive_score.R2LOG];
model = {'GARCH','RealGARCH','RECH'};
prediction_table = table(model',PPS,Violate,Quantile_score,Hit,MSE,MAE,R2LOG);
prediction_table.Properties.VariableNames = {'Model','PPS','Violate','Quantile_score','Hit','MSE','MAE','R2LOG'};
disp(prediction_table)









