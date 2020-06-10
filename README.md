# The codes for Yano, Kaneko, and Komaki


The following pages host the python codes for the paper:

Minimax Predictive Density for Sparse COunt Data, K. Yano, R. Kaneko, and F. Komaki, (under review).

Abstract of the paper:

This paper discusses predictive densities under the Kullback–Leibler loss for high-dimensional Poisson sequence models under sparsity constraints. Sparsity in count data implies zero-inflation. We present a class of Bayes predictive densities that attain exact asymptotic minimaxity in sparse Poisson sequence models. We also show that our class with an estimator of unknown sparsity level plugged-in is adaptive in the exact minimax sense. For application, we extend our results to settings with quasi-sparsity and with missing-completely-at-random observations. The simulation studies as well as application to real data illustrate the efficiency of the proposed Bayes predictive densities.

Class PredDens_IM in PredDens.py offers three methods:
* sample_gen returns a sample from the predictive density
* log_likelihood returns predictive log likelihood
* JointCI_count returns 1 if future observation is contained in the joint coverage set, 0 if not.

Usage (See also Demo.ipynb):
* Import PredDens
* Set optimal scaling in Proposition 2.5
  * scaling_denom = lambda r, kappa: (r**(-kappa))*(1-(r/(r+1))**kappa)/kappa
  * scaling_nom = lambda r: ((r/(r+1))**r)*(1/(r+1))
  * scaling = lambda r, kappa: np.average(scaling_nom(r))/np.average(scaling_denom(r, kappa))
* Set an estimate s_hat of the sparsity level (Default: s_hat = len(np.nonzero(x)[0]))
* Define ins_Proposed = PredDens.ProposedPredictiveDensity(eta=scaling(r, kappa) * s_hat / dim, dim = dim, r =r, x = x, kappa = 0.1)
