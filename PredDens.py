from math import *
import numpy as np
import numpy.random as rd
import scipy.stats as stat
import scipy.special as sp
import scipy.integrate as integrate

########  PredDens.py ####################
#This .py provide predictive densities used in the following paper:
#Paper: Minimax predictive density for sparse count data (under review)
#Authors: Keisuke Yano, Ryoya Kaneko, Fumiyasu Komaki
#Code developed by Ryoya Kaneko, Keisuke Yano, Fumiyasu Komaki


######## Required functions ################

def weighted_L1(v1,v2, r):
    if len(v1) != len(v2) or len(v2) != len(r):
        print("error")
        return -1
    return sum([r[i] * abs(v1[i]-v2[i]) for i in range(len(v1))])


def coverage_count(CI, future):
    c = 0
    for i in range(0, dim):
        if future[i] >= CI[0,i] and future[i] <= CI[1, i]:
            c = c + 1
    return c
    
def delta(a, x):
    if a == x:
        return 1
    else:
        return 0


######## Proposed predictive density ########

class PredDens_IM:
    def __init__(self,  eta , dim, r, x, kappa):
        self.eta = eta  # scalar
        self.dim = dim  #scalar
        self.r = r   # dim vector
        self.x = x  # dim vector 
        self.kappa = kappa #scalar
        
    def sample_gen(self, num=1):
        sample = np.zeros((num, self.dim))     # sample is a (num, dim) vector
        x_ = np.array(self.x)
        r_ = np.array(self.r)
        
        for i in range(0, self.dim):   # index is a parameter which determines the shape of improper functions (larger than 0)
            if isnan(x_[i]) == True:   # if x is nan, return nan 
                sample[:, i] = np.nan
            else:
                if x_[i] != 0.0:
                    w = 0.0
                else:
                    w = 1.0 / (1.0 + self.eta * gamma(self.kappa) / (r_[i] ** self.kappa)) 
                for j in range(0,num):
                    if rd.rand() < w:
                        sample[j, i] = 0
                    else:
                        sample[j, i] = rd.negative_binomial(n = x_[i] + self.kappa, p = r_[i] / (r_[i]+1))                     
        return sample
      
    
                                                            
    def log_likelihood(self, y):    # lengh of y is dim
                                           # index is a parameter which determines the shape of improper functions (larger than 0)
        # exclude coordinates that include nan in either x or y
        x_array = np.array(self.x)
        y_array = np.array(y)
        r_array = np.array(self.r)
        x_ = np.array([x_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        y_ = np.array([y_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        r_ = np.array([r_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        pmf = np.zeros(len(x_))
        
        for i in range(0, len(x_)):
            if x_[i] != 0.0:
                w = 0.0
            else:
                w = 1.0 / (1.0 + self.eta * gamma(self.kappa) / ( r_[i] ** self.kappa)) 
            pmf[i] = w * delta(0, y_[i]) + (1 - w) * stat.nbinom.pmf(y_[i], n = x_[i] + self.kappa, p = r_[i] / (r_[i] + 1))
                                                                         
        return np.nansum(np.array( [np.log(pmf[i]) for i in range(0, len(x_))] ))
    
    
    def CI(self, alpha):  # index is a parameter which determines the shape of improper functions (larger than 0)
        CI_mat = np.zeros((2, self.dim))
        
        x_ = np.array(self.x)
        r_ = np.array(self.r)
        
        for i in range(0, self.dim):
            if isnan(x_[i]) == True:   # if x is nan, return nan 
                CI_mat[:, i] = np.nan
            else:
                if x_[i] != 0.0:
                    w = 0.0
                else:
                    w = 1.0 / (1.0 + self.eta * gamma(self.kappa) / (r_[i] ** self.kappa))                    
                CI_mat[:,i] = w * 0 + (1-w) * np.array(stat.nbinom.interval(alpha, n = x_[i] + self.kappa, p = r_[i] / (r_[i] + 1)))
               
        return CI_mat


    def ww(self):
        r_ = np.array(self.r)
        x_ = np.array(self.x)
        w = np.zeros(self.dim)
        for i in range(self.dim):
            if x_[i] != 0.0:
                w[i] = 0.0
            else:
                w[i] = 1.0 / (1.0 + self.eta * gamma(self.kappa) / (r_[i] ** self.kappa))
                    
                    
        return w
    
    def b_est(self, x):
        theta = np.zeros(self.dim)
        x_ = np.array(x)
        
        for i in range(0, self.dim):
            if x_[i] == 0:
                theta[i] = self.eta  / (self.r[i] ** (self.kappa + 1) + self.r[i] * self.eta)
            else:
                theta[i] = (x_[i] + self.kappa) / self.r[i]
        return theta


######## Competitors  used in the paper ########
        
######## preditive density proposed in Komaki (2004)
class PredDens_K04:
    def __init__(self, beta, dim, r, x):
        self.beta = beta # beta is a vector
        self.dim = dim
        self.r = r  # r is a scalar 
        self.x = x  # size of x is dim
        
    def sample_gen(self, num=1):  # sampling using gamma-mixture expression
        sample = np.zeros((num, self.dim))  #  output is a ndarray with (num, dim) 
        x_ = np.array(self.x)
        sum_x = np.array( [x_[i] for i in range(0, self.dim) if np.isnan(x_[i]) == False] ).sum()

        for i in range(0, self.dim):
            if isnan(x_[i]) == True:
                sample[:, i] = np.nan
            else:
                kappa = np.array([rd.beta(np.sum(self.beta) - 1, sum_x + 1) for i in range(0, num)])
                theta = np.array([ rd.gamma( self.beta[i] + x_[i], ( 1-kappa[j] ) / self.r) for j in range(0, num)])
                sample[:, i] = np.array([rd.poisson(theta[j]) for j in range(0, num)])
        return sample
    
    def log_likelihood(self, y):
        y_array = np.array(y)
        y__ =  [y_array[i] for i in range(0, self.dim) if np.isnan(y_array[i]) == False] 
        x__ = [self.x[i] for i in range(0, self.dim) if np.isnan(y_array[i]) == False]   
        x_ = np.array([x__[i] for i in range(0, len(x__)) if np.isnan(x__[i]) == False]) 
        y_ = np.array([y__[i] for i in range(0, len(y__)) if np.isnan(x__[i]) == False] ) 
        sum_x = np.nansum(x_)
        sum_y = np.nansum(y_)
        
        a = (sum_x + 1) * np.log(self.r / (1 + self.r)) + sum_y * np.log(1 / (1 + self.r))
        b = sp.gammaln(sum_x + sum_y + 1) + sp.gammaln(sum_x + np.sum(self.beta)) \
        - ( sp.gammaln(sum_x + 1) + sp.gammaln(sum_x + sum_y + np.sum(self.beta)))
        c = np.nansum(np.array( [sp.gammaln(x_[i] + y_[i] + self.beta[i]) for i in range(0, len(x_))] ))
        d = np.nansum(np.array( [sp.gammaln(x_[i] + self.beta[i]) for i in range(0, len(x_))] ))
        e = np.nansum(np.array( [np.log(sp.factorial(y_[i])) for i in range(0, len(x_))] ))

        return a + b + c - d - e

    
    
    def CI(self, alpha, N = 10000):
        # coordinate-wise confidence interval with nominal level alpha
        sample = np.zeros((N, self.dim))
        CI_mat = np.zeros((3, self.dim))
        sample = self.sample_gen(num = N)   # sample is an (N, dim) array
        
        for i in range(0, self.dim):
            if isnan(sample[i, 0]) == True:
                CI_mat[:, i] = np.nan
            else:
                CI_mat[0, i] = np.percentile(sample[:, i], (1- alpha) / 2)
                CI_mat[1, i] = np.percentile(sample[:, i], (1+ alpha) / 2)
                CI_mat[2, i] = np.percentile(sample[:, i], 1 / 2)
        return CI_mat
    
    def JointCI_count(self, future, alpha, N):
            sample = self.sample_gen(num = N)
            mean = np.mean(sample, axis = 0)
            dist = [weighted_L1(sample[i,:], mean, self.r * np.ones(self.dim)) for i in range(0, self.dim)]
            a = weighted_L1(future, mean, self.r * np.ones(self.dim))
            if 0 <= a and a <= np.percentile(dist, alpha * 100):
                return 1
            else:
                return 0
            
    def b_est(self, x, num = 10000):
        theta = np.zeros((num, self.dim))
        x_ = np.array(x)
        sum_x = np.sum(x_)
        sum_beta = np.sum(self.beta)
        for i in range(0, self.dim):
            kappa = np.array([rd.beta(sum_beta - 1, sum_x + 1) for i in range(0, num)])
            theta[:, i] = np.array([ rd.gamma( self.beta[i] + x_[i], ( 1-kappa[j] ) / self.r) for j in range(0, num)])
        return np.average(theta, axis = 0)

            
######## preditive density proposed in Komaki (2015)           

def f(u, gamma, x, alpha):
    # used in calculating K
    #u : scalar
    # gamma : n-dim vector
    # x : n-dim vector
    # alpha : scalar

    hoge = [-x[i] * np.log(u / gamma[i] + 1) for i in range(len(x))]
    return u ** (alpha - 1) * np.exp(sum(hoge))

def K(gamma, x, alpha):
    # gamma : n-dim vector
    # x : n-dim vector
    # alpha : scalar
    # h : step size
    def ff(u):
        return f(u, gamma = gamma, x = x, alpha = alpha)
    
    s = integrate.quad(ff, 0, np.inf)
    return s

class PredDens_K15:
    def __init__(self, beta, dim, r, x):
        self.beta = beta
        self.dim = dim
        self.r = r  # r is an n-dim vector 
        self.x = x  # x is an n-dim vector 
                
        
    def sample_gen(self, num = 1):
        sample = np.zeros((num, self.dim))  #  output is a ndarray with (num, dim)        
        x_ = np.array(self.x)
        b_ = np.array(self.beta)
        sum_x = np.array( [x_[i] for i in range(0, self.dim) if np.isnan(x_[i]) == False] ).sum()
        sum_b = sum(b_)
        r_ = np.array(self.r)
        inv_r = [1 / r_[i] for i in range(0, self.dim)]
        
        c = (1 / min(self.r))**(sum_b - 1) * sp.beta(sum_x + 1, sum_b - 1)
        
        for i in range(0, self.dim):
            if isnan(x_[i]) == True:
                sample[:, i] == np.nan

        for i in range(0, num):
            count = 0
            while True:
                count = count + 1
                yy = [rd.negative_binomial(n = x_[i] + b_[i], p = r_[i] / (r_[i] + 1)) for i in range(0, self.dim)]
                u = rd.random()
                if u <= K(inv_r, x_ + yy + b_, sum_b - 1)[0] / c:
                    sample[i, :] = yy
                    break
                if count > 10000:
                    sample[i, :] = np.nan
                    break
        return sample
    
    def log_likelihood(self, y):
        x_ = np.array(self.x)
        y_ = np.array(y)
        b_ = np.array(self.beta)
        sum_x = np.array( [x_[i] for i in range(0, self.dim) if np.isnan(x_[i]) == False] ).sum()
        sum_b = sum(b_)
        r_ = np.array(self.r)
        inv_r = [1 / r_[i] for i in range(0, self.dim)]
        inv_r_plus = [1 / (r_[i]+1) for i in range(0, self.dim)]
        
        a = np.log(K(inv_r, x_ + y_ + b_, sum_b - 1)[0])
        b = np.log(K(inv_r_plus, x_ + b_, sum_b - 1)[0])
        c = [ np.log(stat.nbinom.pmf(y_[i], n = x_[i] + b_[i], p = r_[i] / (r_[i] + 1))) for i in range(0, len(y)) ]
        return sum(c) + a - b


    
    
######## poisson likelihood maximization with l1 regularization
class PredDens_L1:
    
     # coordinate-wise
    
    def __init__(self, lamb, dim, r, x):
        self.lamb = lamb   # regularization parameter (size is dim )
        self.dim = dim   
        self.r = r     # size of r is dim
        self.x = x    # size of x is dim
        
    def sample_gen(self, num = 1):
        sample = np.zeros((num, self.dim))
        x_ = np.array(self.x)
        r_ = np.array(self.r)
        lamb_ = np.array(self.lamb)
        
        for i in range(0, self.dim):
            if np.isnan(x_[i]) == True:
                sample[:,i] = np.nan
            else:
                theta = x_[i] / (r_[i] - lamb_[i])
                sample[:,i] = rd.poisson(lam = theta, size = num)
        return sample
    
    def log_likelihood(self, y):
        x_array = np.array(self.x)
        y_array = np.array(y)
        r_array = np.array(self.r)
        lamb_array = np.array(self.lamb)
        x_ = np.array([x_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        y_ = np.array([y_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        r_ = np.array([r_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        lamb_ = np.array([lamb_array[i] for i in range(0, self.dim) if np.isnan(x_array[i]) == False and np.isnan(y_array[i]) == False])
        pmf = np.zeros(len(x_))

        for i in range(0, len(x_)):
            pmf[i] = stat.poisson.pmf(y[i], mu = x_[i] / (r_[i] - lamb_[i])) 
        
        return np.array( [np.log(pmf[i]) for i in range(0, len(x_))] ).sum()
    
    def CI(self, alpha):
        CI_mat = np.zeros((2, self.dim))
        
        x_ = np.array(self.x)
        r_ = np.array(self.r)
        lamb_ = np.array(self.lamb)
        
        for i in range(0, self.dim):
            if isnan(x_[i]) == True:   # if x is nan, return nan 
                CI_mat[:, i] = np.nan
            else:
                CI_mat[:, i] = np.array(stat.poisson.interval(alpha, mu = x_[i] / (r_[i] - lamb_[i]) ))
        return CI_mat
    
    def JointCI_count(self, future, alpha, N):
            sample = self.sample_gen(num = N)
            mean = np.mean(sample, axis = 0)
            dist = [weighted_L1(sample[i,:], mean, self.r) for i in range(0, self.dim)]
            a = weighted_L1(future, mean, self.r)
            if 0 <= a and a <= np.percentile(dist, alpha * 100):
                return 1
            else:
                return 0
            
    def l1_est(self, x):
        x_ = np.array(x)
        theta = np.zeros(self.dim)
        for i in range(0, self.dim):
            theta = x_[i] / (self.r[i] - self.lamb[i])
        return theta