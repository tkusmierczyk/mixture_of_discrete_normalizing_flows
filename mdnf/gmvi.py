# -*- coding: utf-8 -*-
""" Variational Gaussian Mixture using Discrete Normalizing Flows. """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import logging
logger = logging.getLogger(__name__)

from prml.rv import VariationalGaussianMixture

from time_profiling import timing

import time
import collections
import sklearn
import sklearn.cluster
import sklearn.metrics
import scipy
from scipy.special import digamma, gamma, logsumexp, loggamma, multigammaln



def minimal_matching(X1, X2):
    """ Returns a vector with numbers of rows from X2 matching rows from X1. 

        Can be used to find what's the best mapping (relabeling) between various HMMs.
    """
    costs = sklearn.metrics.pairwise.euclidean_distances(X1, X2)
    rows, shuffling = scipy.optimize.linear_sum_assignment(costs)
    assert np.alltrue(rows==np.array(range(len(rows))))
    return shuffling


def _assert_valid(v, name=""):
    assert tf.reduce_any(tf.math.is_nan(v))==False, "%s contains NaNs:\n%s" % (name, v)
    minv = tf.reduce_min(v)
    assert minv>=0.0, "min(%s)=%s\n%s" % (name, minv, v)
    maxv = tf.reduce_max(v)
    assert maxv<=1.00001, "max(%s)=%s\n%s" % (name, maxv, v)


def yield_indices1(x_train, batch_size, drop_remainder=False):
    """ Generates indices of elements in minibatches. """
    data_ixs = np.arange(x_train.shape[0])
    np.random.shuffle(data_ixs)
    d = tf.data.Dataset.from_tensor_slices(data_ixs).batch(batch_size, drop_remainder)
    for ixs in d:
        yield ixs

def yield_indices(x_train, batch_size, drop_remainder=False):
    """ Generates indices of elements in minibatches in an infinite loop. """
    while True:        
        for ixs in yield_indices1(x_train, batch_size, drop_remainder):
            yield ixs

def log_C_func(alpha):
    """ Bishop's book Eq. B.23 """
    return loggamma(np.sum(alpha)) - np.sum(loggamma(alpha)) 


def log_B_func(W, dof): 
    """ Bishop's book Eq. B.79 """
    D = W.shape[-1]
    log_part1 = -dof/2. * np.linalg.slogdet(W)[1]
    log_part2 = -(dof*D/2.)*np.log(2.) -multigammaln(dof*0.5, D)
    return log_part1 + log_part2


def entropy_wishart(W, dof): 
    """ Bishop's book Eq. B.82 """
    D = W.shape[-1]
    expectation = digamma(0.5 * (dof - np.arange(D)[:, None])).sum(axis=0) \
                        + D * np.log(2.) + np.linalg.slogdet(W)[1] # B.81
    return -log_B_func(W, dof) - (dof-D-1.)/2. * expectation + dof*D/2.




class VariationalGaussianMixtureELBO(VariationalGaussianMixture):
    """ Variational Gaussian Mixture equipped with evaluation of ELBO. 

        Based on the code from https://github.com/ctgk/PRML. 
    """
    
    def __init__(self,
        n_components=1,
        alpha0=None,
        m0=None,
        W0=1.0,
        dof0=None,
        beta0=1.0,
        initialization='random',
        elbo_nsamples=1000):      
        """
            Args:
                elbo_nsamples  How many samples to use to evaluate ELBO.
        """  
        super().__init__(n_components=n_components, 
                         alpha0=alpha0, 
                         m0=m0, 
                         W0=W0, 
                         dof0=dof0, 
                         beta0=beta0)
        self.elbo_nsamples = elbo_nsamples
        self.initialization = initialization


    def _init_params(self, X):
        sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_components) * self.alpha0
        if self.m0 is None:
            self.m0 = np.mean(X, axis=0)
        else:
            self.m0 = np.zeros(self.ndim) + self.m0
        self.W0 = np.eye(self.ndim) * self.W0
        if self.dof0 is None:
            self.dof0 = self.ndim

        self.component_size = sample_size / self.n_components + np.zeros(self.n_components)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(sample_size, self.n_components, replace=False)
        self.mu = X[indices]
        self.W = np.tile(self.W0, (self.n_components, 1, 1))
        self.dof = self.dof0 + self.component_size

    def _variational_maximization(self, X, r):
        self.component_size = r.sum(axis=0)
        Xm = (X.T.dot(r) / self.component_size).T
        d = X[:, None, :] - Xm
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.component_size[:, None, None]
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.mu = (self.beta0 * self.m0 + self.component_size[:, None] * Xm) / self.beta[:, None]
        d = Xm - self.m0
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S.T).T
            + (self.beta0 * self.component_size * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.component_size)).T)
        self.dof = self.dof0 + self.component_size

    def _variational_expectation(self, X):
        d = X[:, None, :] - self.mu
        maha_sq = -0.5 * (
            self.ndim / self.beta
            + self.dof * np.sum(
                np.einsum("kij,nkj->nki", self.W, d) * d, axis=-1))
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        ln_r = ln_pi + 0.5 * ln_Lambda + maha_sq
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r)
        return r
        
    def variational_expectation(self, X, e_batch_size=1024, e_batch_drop_remainder=False, **kwargs):
        for i, minibatch_ixs in enumerate( yield_indices1(X, e_batch_size, e_batch_drop_remainder) ):
            logger.debug("[VGM-ELBO.variational_expectation] minibatch_no=%d size=%d" % (i, len(minibatch_ixs)) )
            self.r[minibatch_ixs] = self._variational_expectation(X[minibatch_ixs])
        return self.r
        
    def sample_posteriors_global(self, nsamples=1000):
        posterior_pi = tfd.Dirichlet(self.alpha)
        pis = posterior_pi.sample(nsamples)

        W1 = np.tile(self.W, [nsamples,1,1,1]) #nsamples x K x N=2 x N=2
        posterior_lambda = tfd.WishartTriL(df=self.dof, 
                                           scale_tril=tf.linalg.cholesky(W1))
        lambdas = posterior_lambda.sample() #nsamples x K x N=2 x N=2

        def get_posterior_mu(beta, mu, lambdas):
            locations = np.broadcast_to(mu, lambdas.shape[0:1]+mu.shape)
            precisions = (lambdas*beta[None,:,None,None])
            covs = tf.linalg.inv(precisions) #!
            covs = 0.5*(covs + tf.transpose(covs, [0, 1, 3, 2])) # numerical stability workaround
            d = tfd.MultivariateNormalTriL(loc=locations, 
                                           scale_tril=tf.linalg.cholesky(covs))
            return d
        posterior_mu = get_posterior_mu(self.beta, 
                                          self.mu, 
                                          lambdas)
        mus = posterior_mu.sample()

        return posterior_pi, posterior_lambda, posterior_mu, pis, lambdas, mus

    def sample_posterior_zs(self, r, nsamples=1000):
        posterior_z = tfd.OneHotCategorical(probs=r, dtype=r.dtype)
        zs = posterior_z.sample(nsamples) 
        return posterior_z, zs      
        
    @staticmethod
    def get_norm_log_probs(mus, lambdas, X):
        ilambdas = tf.linalg.inv(lambdas)
        ilambdas = 0.5*(ilambdas + tf.transpose(ilambdas, [0, 1, 3, 2])) # numerical stability workaround
        d = tfd.MultivariateNormalTriL(loc=mus, 
                        scale_tril=tf.linalg.cholesky(ilambdas))
        x_log_probs = [d.log_prob(X[n]) for n in range(X.shape[0])]
        return tf.stack(x_log_probs, 1)     

    def E_log_p_X(self, X, r): # Eq. 10.71
        """ E( log p(X|Z, mu, lambda) ) """
        # 10.51
        component_size = r.sum(axis=0)

        # 10.52 -> K x D 
        Xm = (X.T.dot(r) / component_size).T

        # 10.53
        d = X[:, None, :] - Xm
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / component_size[:, None, None] # K x D x D      

        # 10.65 -> shape = K
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) \
                    + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]

        # Eq. 10.71:
        part2 = -self.ndim/self.beta
        part3 = -self.dof * np.trace(np.matmul(S, self.W), axis1=1, axis2=2) # 
        d = Xm - self.mu # K x D
        part4 = -self.dof * ( np.matmul(d[:,None,:], self.W)[:,0,:] * d ).sum(-1) # TODO nicer implementation!
        part5 = -self.ndim * np.log(2*np.pi)
        return 0.5*np.sum( component_size * (ln_Lambda + part2 + part3 + part4 + part5) ) 

    def E_log_p_X_samples_z(self, X, zs):
        """ Estimate differentiable w.r.t draws ~Z. """
        # allocations from samples
        r_tf = tf.reduce_mean(zs, 0) + 1e-16
    
        # 10.51
        component_size_tf = tf.reduce_sum(r_tf, 0)

        # 10.52 -> K x D 
        Xm_tf = tf.reduce_sum(tf.transpose(r_tf)[...,None]*X, axis=-2) / component_size_tf[...,None]

        # 10.53
        d = X[:, None, :] - Xm_tf
        S_tf = tf.reduce_sum(r_tf[...,None,None] * (d[...,None] * d[...,None,:]), 0) / component_size_tf[...,None,None]

        # 10.65 -> shape = K
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) \
                    + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]

        # Eq. 10.71:
        part2 = -self.ndim/self.beta        
        part3_tf = -self.dof * tf.linalg.trace(tf.matmul(S_tf, self.W))
        d = Xm_tf - self.mu # K x D
        part4_tf = -self.dof * tf.reduce_sum(tf.matmul(d[:,None], self.W)[:,0,:]*d, -1)
        part5 = -self.ndim * np.log(2*np.pi)
        return 0.5*tf.reduce_sum( component_size_tf * (ln_Lambda + part2 + part3_tf + part4_tf + part5) ) 

    def E_log_p_X_samples(self, X, mus, lambdas, zs):
        # E( log p(X|Z, mu, lambda) ):
        x_log_probs = VariationalGaussianMixtureELBO.get_norm_log_probs(mus, lambdas, X)
        lik = tf.reduce_sum(zs * x_log_probs, [-1,-2]) # sum over data samples and components
        return tf.reduce_mean(lik)    

    def E_log_p_Z(self, r): # 10.72     
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum()) # 10.66
        return np.sum(r*ln_pi) 

    def E_log_p_Z_samples_z(self, zs):
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum()) # 10.66        
        r_tf = tf.reduce_mean(zs, 0) # allocations from samples
        return tf.reduce_sum(r_tf*ln_pi) 
                
    def E_log_p_Z_samples(self, pis, zs):
        log_prob_zs = tf.reduce_sum( zs * tf.math.log(pis[:,None,:]), [-1,-2]) 
        return tf.reduce_mean(log_prob_zs) 
        
    def E_log_p_pi(self): # 10.73:
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum()) # 10.66
        return log_C_func(self.alpha0) + np.sum( (self.alpha0-1)*ln_pi )

    def E_log_p_pi_samples(self, pis):
        prior_pi = tfd.Dirichlet(self.alpha0)
        log_prior_pis = prior_pi.log_prob(pis)
        return tf.reduce_mean(log_prior_pis)

    def E_log_p_mu_lambda(self): # 10.74
        # 10.65 -> shape = K
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) \
                    + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        
        # 10.74: 
        part11 = self.ndim * np.log(self.beta0 / (2.*np.pi))    
        part13 = -self.ndim * self.beta0 / self.beta
        d = self.mu - self.m0
        part14 = -self.beta0*self.dof * ( np.matmul(d[:,None,:], self.W)[:,0,:] * d ).sum(-1) # TODO ugly!
        part1 = 0.5 * np.sum(part11 + ln_Lambda + part13 + part14)
        
        part2 = self.n_components * log_B_func(self.W0, self.dof0)
        part3 = (self.dof0-self.ndim-1)/2. * np.sum(ln_Lambda)
        part4 = -0.5 * np.sum( self.dof * np.trace( np.matmul(np.linalg.inv(self.W0), self.W), 
                                                    axis1=-2, axis2=-1) )                
        return part1 + part2 + part3 + part4

    def E_log_p_mu_lambda_samples(self, mus, lambdas):
        prior_lambda = tfd.WishartTriL(df=self.dof0, #, 
                                       scale_tril=tf.linalg.cholesky(self.W0))
        log_prior_lambdas = tf.reduce_sum( prior_lambda.log_prob(lambdas), -1 )
        
        def get_prior_mu(beta0, m0, lambdas):
            precisions = (lambdas*beta0)
            covs = tf.linalg.inv(precisions)
            covs = 0.5*(covs + tf.transpose(covs, [0, 1, 3, 2])) # numerical stability workaround
            d = tfd.MultivariateNormalTriL(loc=m0, 
                                           scale_tril=tf.linalg.cholesky(covs))
            return d
        prior_mu = get_prior_mu(self.beta0, #, 
                                self.m0, 
                                lambdas)
        log_prior_mus = tf.reduce_sum( prior_mu.log_prob(mus), -1)  
        return tf.reduce_mean(log_prior_lambdas+log_prior_mus)

    def E_log_q_Z(self, r): # 10.75
        return np.sum( r * np.log(r+1e-61) )

    def E_log_q_pi(self): # 10.76:
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum()) # 10.66
        return log_C_func(self.alpha) + np.sum( (self.alpha-1)*ln_pi )        

    def E_log_q_mu_lambda(self): # 10.77
        # 10.65 -> shape = K
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) \
                            + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        return np.sum(  0.5*ln_Lambda 
                        +self.ndim/2.*np.log(self.beta/(2.*np.pi)) 
                        -self.ndim/2. 
                        -entropy_wishart(self.W, self.dof) )  

    def elbo_cf(self, X, r, minibatch_scaling):
        """ Closed-form ELBO evaluation. """
        E_log_p_pi = self.E_log_p_pi()
        E_log_p_mu_lambda = self.E_log_p_mu_lambda()
        E_log_p_Z = self.E_log_p_Z(r)
        E_log_p_X = self.E_log_p_X(X, r)
        E_log_q_pi =  self.E_log_q_pi()
        E_log_q_mu_lambda = self.E_log_q_mu_lambda()
        E_log_q_Z = self.E_log_q_Z(r)
        logger.debug("[VGM-ELBO-CF] E(log p(pi))=%.3f E(log p(mu,lam))=%.3f E(log p(Z|.))=%.3f E(log p(X|.))=%.3f" % (
                     E_log_p_pi, E_log_p_mu_lambda, E_log_p_Z, E_log_p_X))        
        logger.debug("[VGM-ELBO-CF] E(log q(pi))=%.3f E(log q(mu,lam))=%.3f E(log q(Z))=%.3f" % (
                          E_log_q_pi, E_log_q_mu_lambda, E_log_q_Z))
       
        joint_log_prob = E_log_p_pi+E_log_p_mu_lambda+E_log_p_Z+E_log_p_X        
        entropy_global = -E_log_q_pi -E_log_q_mu_lambda      
        entropy_Z = -E_log_q_Z
        elbo_estimate = joint_log_prob*minibatch_scaling + entropy_global + entropy_Z*minibatch_scaling
        logger.debug("[VGM-ELBO-CF] ELBO~%.2f (jnt=%.2f*%.1f e1=%.2f e2=%.2f*%.1f) on batch=%d" % (
                     elbo_estimate, joint_log_prob, minibatch_scaling, entropy_global, entropy_Z, minibatch_scaling, len(X)))
        return elbo_estimate, E_log_p_X, E_log_p_Z, E_log_p_pi+E_log_p_mu_lambda, entropy_global, entropy_Z

    def elbo_samples_z(self, X, r, minibatch_scaling):
        """ ELBO hybrid (allocations are sampled) evaluation. """
        posterior_z, zs = self.sample_posterior_zs(r, self.elbo_nsamples)        

        E_log_p_pi = self.E_log_p_pi()
        E_log_p_mu_lambda = self.E_log_p_mu_lambda()
        E_log_p_Z = self.E_log_p_Z_samples_z(zs)
        E_log_p_X = self.E_log_p_X_samples_z(X, zs)
        E_log_q_pi = self.E_log_q_pi()
        E_log_q_mu_lambda = self.E_log_q_mu_lambda()
        r_tf = tf.reduce_mean(zs, 0) # restore allocations from samples
        E_log_q_Z = tf.reduce_sum( r_tf*tf.math.log(r_tf+1e-61) ) # and use exact formula

        logger.debug("[VGM-ELBO-MZ] E(log p(pi))=%.3f E(log p(mu,lam))=%.3f E(log p(Z|.))=%.3f E(log p(X|.))=%.3f" % (
                     E_log_p_pi, E_log_p_mu_lambda, E_log_p_Z, E_log_p_X))        
        logger.debug("[VGM-ELBO-MZ] E(log q(pi))=%.3f E(log q(mu,lam))=%.3f E(log q(Z))=%.3f" % (
                          E_log_q_pi, E_log_q_mu_lambda, E_log_q_Z))
       
        joint_log_prob = E_log_p_pi+E_log_p_mu_lambda+E_log_p_Z+E_log_p_X        
        entropy_global = -E_log_q_pi -E_log_q_mu_lambda      
        entropy_Z = -E_log_q_Z
        elbo_estimate = joint_log_prob*minibatch_scaling + entropy_global + entropy_Z*minibatch_scaling
        logger.debug("[VGM-ELBO-MZ] ELBO~%.2f (jnt=%.2f*%.1f e1=%.2f e2=%.2f*%.1f) on batch=%d" % (
                     elbo_estimate, joint_log_prob, minibatch_scaling, entropy_global, entropy_Z, minibatch_scaling, len(X)))
        return elbo_estimate, E_log_p_X, E_log_p_Z, E_log_p_pi+E_log_p_mu_lambda, entropy_global, entropy_Z
                                                       
    def elbo_samples(self, X, r, minibatch_scaling):
        """ MC ELBO estimation. """
        posterior_pi, posterior_lambda, posterior_mu, pis, lambdas, mus = \
                                                self.sample_posteriors_global(self.elbo_nsamples)
        posterior_z, zs = self.sample_posterior_zs(r, self.elbo_nsamples)

        E_log_p_pi = self.E_log_p_pi_samples(pis)
        E_log_p_mu_lambda = self.E_log_p_mu_lambda_samples(mus, lambdas)
        E_log_p_Z = self.E_log_p_Z_samples(pis, zs)
        E_log_p_X = self.E_log_p_X_samples(X, mus, lambdas, zs)
        E_log_q_pi = posterior_pi.log_prob(pis)
        E_log_q_pi = tf.reduce_mean(E_log_q_pi) # average over samles
        E_log_q_mu_lambda = tf.reduce_sum( posterior_mu.log_prob(mus), -1) + \
                            tf.reduce_sum( posterior_lambda.log_prob(lambdas), -1)
        E_log_q_mu_lambda = tf.reduce_mean(E_log_q_mu_lambda) # average over samples
        r_tf = tf.reduce_mean(zs, 0) # restore allocations from samples
        E_log_q_Z = tf.reduce_sum( r_tf*tf.math.log(r_tf+1e-61) ) # and use exact formula

        logger.debug("[VGM-ELBO-MC] E(log p(pi))=%.3f E(log p(mu,lam))=%.3f E(log p(Z|.))=%.3f E(log p(X|.))=%.3f" % (
                     E_log_p_pi, E_log_p_mu_lambda, E_log_p_Z, E_log_p_X))        
        logger.debug("[VGM-ELBO-MC] E(log q(pi))=%.3f E(log q(mu,lam))=%.3f E(log q(Z))=%.3f" % (
                          E_log_q_pi, E_log_q_mu_lambda, E_log_q_Z))
       
        joint_log_prob = E_log_p_pi+E_log_p_mu_lambda+E_log_p_Z+E_log_p_X        
        entropy_global = -E_log_q_pi -E_log_q_mu_lambda      
        entropy_Z = -E_log_q_Z
        elbo_estimate = joint_log_prob*minibatch_scaling + entropy_global + entropy_Z*minibatch_scaling
        logger.debug("[VGM-ELBO-MC] ELBO~%.2f (jnt=%.2f*%.1f e1=%.2f e2=%.2f*%.1f) on batch=%d" % (
                     elbo_estimate, joint_log_prob, minibatch_scaling, entropy_global, entropy_Z, minibatch_scaling, len(X)))
        return elbo_estimate, E_log_p_X, E_log_p_Z, E_log_p_pi+E_log_p_mu_lambda, entropy_global, entropy_Z

    def elbo(self, x_train, batch_size=10000):
        total_p_X, total_p_Z, total_p_priors, total_entropy_Z = 0., 0., 0., 0.
        for minibatch_ixs in yield_indices1(x_train, batch_size, drop_remainder=False):
            minibatch_x_train, minibatch_r = x_train[minibatch_ixs], self.r[minibatch_ixs]
            minibatch_scaling = float(x_train.shape[0])/minibatch_x_train.shape[0]            
            
            #self.elbo_samples(minibatch_x_train, minibatch_r, minibatch_scaling) # compare with MC version
            #self.elbo_samples_z(minibatch_x_train, minibatch_r, minibatch_scaling) # compare with hybrid-MC version
            _, E_log_p_X, E_log_p_Z, E_log_p_priors, entropy_global, entropy_Z = \
                self.elbo_cf(minibatch_x_train, minibatch_r, minibatch_scaling)         

            total_p_X += E_log_p_X
            total_p_Z += E_log_p_Z
            total_entropy_Z += entropy_Z
        return total_p_X+total_p_Z+total_entropy_Z+E_log_p_priors+entropy_global

    def _init_allocations(self, x_train):
        if self.initialization=='random':
            logger.debug("[VGM-ELBO.fit] Random initialization")            
            r = np.random.random( (x_train.shape[0], self.n_components) )
            self.r = r/r.sum(1,keepdims=True)
        elif self.initialization=='kmeans':
            logger.debug("[VGM-ELBO.fit] Initialization with kmeans")
            kmeans = sklearn.cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(x_train)
            clusters = kmeans.predict(x_train)
            logger.debug("[VGM-ELBO.fit] initiali clusters' allocation=%s" % collections.Counter(clusters))

            eps = 1e-2
            r = np.ones( (len(clusters), self.n_components), dtype=float) * eps/(self.n_components-1)
            r[range(len(clusters)), clusters]= 1-eps
            self.r = r/r.sum(1,keepdims=True)

            logger.debug("[VGM-ELBO.fit] Performing zero M-step.")
            self._variational_maximization(x_train, self.r)    
        else:
            raise ValueError("Wrong initialization type: %s! Try random or kmeans." % self.initialization)    

    def fit(self, x_train, 
            min_niter=0, max_niter=100, noimprov_niter=20, 
            batch_size=512,      
            callback_iter=lambda obj, iter, elbo: \
                logger.info("[VGM-ELBO.fit] iter=%s ELBO=%.2f" % (iter, elbo)),
            continue_training=False,
            **kwargs):
        """
            Args:
                **kwargs  Will be passed to variational_expectation (E-step).
        """  
        if not continue_training:
            self._init_params(x_train)
            self._init_allocations(x_train)
            elbo = self.elbo(x_train, batch_size=batch_size)
            callback_iter(self, -1, elbo)

        params = np.hstack([param.flatten() for param in self.get_params()])
        best_elbo, best_r, last_improvement = -float("inf"), None, 0
        for j in range(max_niter):
            logger.debug("[VGM-ELBO.fit] E-step")
            self.variational_expectation(x_train, step_no=j, **kwargs)              

            logger.debug("[VGM-ELBO.fit] M-step")
            self._variational_maximization(x_train, self.r)

            elbo = self.elbo(x_train, batch_size=batch_size)
            callback_iter(self, j, elbo)
                        
            if elbo>best_elbo:
                best_elbo = elbo
                best_r = np.array(self.r)
                last_improvement = j

            if j>=min_niter and j-last_improvement >= noimprov_niter:
                logger.info("[VGM-ELBO.fit] Converged due to no improvement in last %i iterations." % noimprov_niter)
                break            
            new_params = np.hstack([param.flatten() for param in self.get_params()])                    
            if j>=min_niter and np.allclose(new_params, params): 
                logger.info("[VGM-ELBO.fit] Converged due to no change in parameters.")
                break
            else: params = np.copy(new_params)
        self.r = best_r
        return best_r
    

class VariationalGaussianMixtureFlows(VariationalGaussianMixtureELBO):
    """ Inference with E-step performed by ELBO optimization w.r.t. params of approximating flows. """
    
    def __init__(self, inference, **kwargs):        
        super().__init__(**kwargs)          
        self.inference = inference
    """
        Args:
            inference  Inference object performing variational E-step.
                       For example, VariationalInference or BoostingVariationalInference.
    """

    @property
    def temperature(self):
        return self.inference.temperature

    @timing
    def variational_expectation(self, X, step_no, 
                                e_batch_size=512, e_batch_drop_remainder=False,
                                temperature_annealing=lambda step_no, iteration: 0.2):               

        # priors and entropies of continous variables
        # entropy of Z is added in inference object
        E_log_p_pi = self.E_log_p_pi()
        E_log_p_mu_lambda = self.E_log_p_mu_lambda()
        E_log_q_pi = self.E_log_q_pi()
        E_log_q_mu_lambda = self.E_log_q_mu_lambda()
       
        # model log of joint probability
        minibatch_ixs_generator = yield_indices(X, e_batch_size, e_batch_drop_remainder)
        self.minibatch_status = {"len": "?", "scaling": float('nan')} # to be used also in callback_log
        def log_joint_probs(zs):
            #zs = wrap_categories1(zs, self.n_components)             

            minibatch_ixs = next(minibatch_ixs_generator)
            minibatch_x, minibatch_zs = X[minibatch_ixs], tf.gather(zs, minibatch_ixs, axis=-2)
            minibatch_scaling = float(X.shape[0]) / minibatch_x.shape[0]

            self.minibatch_status["len"] = len(minibatch_x)
            self.minibatch_status["scaling"] = minibatch_scaling

            E_log_p_Z = self.E_log_p_Z_samples_z(minibatch_zs)
            E_log_p_X = self.E_log_p_X_samples_z(minibatch_x, minibatch_zs)

            return (E_log_p_Z+E_log_p_X)*minibatch_scaling + E_log_p_pi + E_log_p_mu_lambda

        # fix inference object properties
        self.inference.log_prob = lambda zs: tf.cast( 
                log_joint_probs(tf.cast(zs, X.dtype))-E_log_q_pi-E_log_q_mu_lambda,  
                                                      self.inference.dtype)
        self.inference.temperature_annealing = lambda iteration: temperature_annealing(step_no, iteration)

        # fit using inference object
        start_time = time.time()               
        def callback_log(status, iteration, loss):
            #if iteration%10!=0: return
            logger.debug("[VGMF.Estep][%.2fs] step=%s. iter=%i. ELBO=%.2f t=%s e_batch=%s*%.1f nsamples=%s" % \
                        (time.time()-start_time, step_no, iteration, -loss, status.temperature, 
                         self.minibatch_status["len"],  self.minibatch_status["scaling"], self.inference.nsamples) )
        self.inference.fit(callback=callback_log)    

        # apply fit best solutions  
        self.inference.flow = self.inference.best_flow
        self.inference.base = self.inference.best_base

        # estimate posterior expectations
        self._update_allocations(X.dtype)

        return self.r

    def _update_allocations(self, dtype):
        self.r = 1e-24
        for i in range(10):
            base_samples, mask = self.inference.base.sample_extm(self.inference.nsamples) 
            zs = self.inference.flow(base_samples, mask)               
            #zs = wrap_categories1(zs, self.n_components)         
            self.r += tf.cast(tf.reduce_mean(zs, 0), dtype).numpy()
        self.r = self.r / np.sum(self.r, -1, keepdims=True)           

    def _update_allocations0(self, dtype, min_nsamples=1000): # legacy version
        nsamples = max(2 * self.inference.nsamples, min_nsamples) 
        base_samples, mask = self.inference.base.sample_extm(nsamples) 
        zs = self.inference.flow(base_samples, mask)                        
        #zs = wrap_categories1(zs, self.n_components)
        self.r = tf.cast(tf.reduce_mean(zs, 0), dtype)
        self.r = self.r+1e-24 #!
        self.r = self.r.numpy()

    def fit(self, x_train, 
            min_niter=0, max_niter=100, noimprov_niter=20, 
            batch_size=512, 
            callback_iter=lambda obj, iter, elbo: \
                            logger.info("[VGMF.fit] iter=%s ELBO=%.2f" % (iter, elbo)),
            continue_training=False,
            **kwargs):  
        """
            Args:
                **kwargs  Will be passed to variational_expectation (E-step).
        """          
        return super().fit(x_train, 
                            min_niter=min_niter, max_niter=max_niter, noimprov_niter=noimprov_niter, 
                            batch_size=batch_size, 
                            continue_training=continue_training, callback_iter=callback_iter,
                            **kwargs)
                        

class VariationalGaussianMixtureFlowsSamples(VariationalGaussianMixtureFlows):
    """ Estimates E_log_p_Z and E_log_p_X only from samples. """

    @timing
    def variational_expectation(self, X, step_no, 
                                e_batch_size=512, e_batch_drop_remainder=False,
                                temperature_annealing=lambda step_no, iteration: 0.2):

        posterior_pi, posterior_lambda, posterior_mu, pis, lambdas, mus = \
                                                self.sample_posteriors_global(self.inference.nsamples)
                
        # priors and entropies of continous variables
        # entropy of Z is added in inference object
        E_log_p_pi = self.E_log_p_pi()
        E_log_p_mu_lambda = self.E_log_p_mu_lambda()
        E_log_q_pi = self.E_log_q_pi()
        E_log_q_mu_lambda = self.E_log_q_mu_lambda()
       
        # model log of joint probability
        minibatch_ixs_generator = yield_indices(X, e_batch_size, e_batch_drop_remainder)
        self.minibatch_status = {"len": "?", "scaling": float('nan')} # to be used also in callback_log
        def log_joint_probs(zs):
            #zs = wrap_categories1(zs, self.n_components)

            minibatch_ixs = next(minibatch_ixs_generator)
            minibatch_x, minibatch_zs = X[minibatch_ixs], tf.gather(zs, minibatch_ixs, axis=-2)
            minibatch_scaling = float(X.shape[0]) / minibatch_x.shape[0]

            self.minibatch_status["len"] = len(minibatch_x)
            self.minibatch_status["scaling"] = minibatch_scaling

            E_log_p_Z = self.E_log_p_Z_samples(pis, zs) #!
            E_log_p_X = self.E_log_p_X_samples(X, mus, lambdas, zs) #!

            return (E_log_p_Z+E_log_p_X)*minibatch_scaling + E_log_p_pi + E_log_p_mu_lambda

        # fix inference object properties
        self.inference.log_prob = lambda zs: tf.cast( 
                                    log_joint_probs(tf.cast(zs, X.dtype))-E_log_q_pi-E_log_q_mu_lambda,  
                                                      self.inference.dtype)
        self.inference.temperature_annealing = lambda iteration: temperature_annealing(step_no, iteration)

        # fit using inference object
        start_time = time.time()               
        def callback_log(status, iteration, loss):
            #if iteration%10!=0: return
            logger.debug("[VGMF.Estep][%.2fs] step=%s. iter=%i. ELBO=%.2f t=%s e_batch=%s*%.1f nsamples=%s" % \
                        (time.time()-start_time, step_no, iteration, -loss, status.temperature, 
                         self.minibatch_status["len"],  self.minibatch_status["scaling"], self.inference.nsamples) )
        self.inference.fit(callback=callback_log)    

        # apply fit best solutions  
        self.inference.flow = self.inference.best_flow
        self.inference.base = self.inference.best_base

        # estimate posterior expectations
        self._update_allocations(X.dtype)
        return self.r

