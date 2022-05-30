# RECH and RealRECH
This repository includes the code that implements Bayesian inference and forecasting for the RECH and RealRECH models.

The folders SMC_for_GARCH, SMC_for_RealGARCH and SMC_for_RECH contain the implementation of SMC algorithm for fitting GARCH, RealGARCH and RECH (with realized measure) respectively.

Run SP500_analysis.m to reproduce the empirical results for SP500 data.

Please, if you use the code in your published work, cite us as:

@article{https://doi.org/10.1002/jae.2902,
author = {Nguyen, T.-N. and Tran, M.-N. and Kohn, R.},
title = {Recurrent Conditional Heteroskedasticity†},
journal = {Journal of Applied Econometrics},
doi = {https://doi.org/10.1002/jae.2902},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.2902},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.2902},
abstract = {Summary We propose a new class of financial volatility models, called the REcurrent Conditional Heteroskedastic (RECH) models, to improve both in-sample analysis and out-of-sample forecasting of the traditional conditional heteroskedastic models. In particular, we incorporate auxiliary deterministic processes, governed by recurrent neural networks, into the conditional variance of the traditional conditional heteroskedastic models, e.g. GARCH-type models, to flexibly capture the dynamics of the underlying volatility. RECH models can detect interesting effects in financial volatility overlooked by the existing conditional heteroskedastic models such as the GARCH, GJR and EGARCH. The new models often have good out-of-sample forecasts while still explaining well the stylized facts of financial volatility by retaining the well-established features of econometric GARCH-type models. These properties are illustrated through simulation studies and applications to thirty-one stock indices and exchange rate data. An user-friendly software package, together with the examples reported in the paper, is available at https://github.com/vbayeslab.}
}
