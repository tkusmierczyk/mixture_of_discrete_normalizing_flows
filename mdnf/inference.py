# -*- coding: utf-8 -*-
""" Variational inference algorithms for discrete normalizing flows. """

import numpy as np
import tensorflow as tf  

import copy

import time
from collections import Counter

import logging
logger = logging.getLogger(__name__)

import prob_recovery
import cardinality



class TemperatureAnnealingExp():
    """ Temperature annealing schedule. """

    def __init__(self, base_temp=0.1, anneal_rate=0.001, min_temp=0.001):
        self.base_temp = base_temp
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp

    def __call__(self, iteration):
        t = self.base_temp * np.exp(-self.anneal_rate*iteration)
        return max(t, self.min_temp)   


# Ignore loss improvements smaller than this
IMPROV_EPS = 1e-4


def approximate_entropy_from_samples(sample, cardinalities=None):
    """ Estimates entropy by counting outcomes (non-differentiable).

        If no independence between dimensions/variables can be assumed 
        then each outcome needs to be counted separately.
        Approximated by using only finite number of samples and 
        by hashing outcomes (only when NxK > 59).
    """
    sample = cardinality.wrap_categories(sample, cardinalities)
    sample_flattened = tf.reshape(sample, (sample.shape[0], -1)) 

    position_weights = np.array([pow(2, v, 909090909090909091) for v in range(np.prod(sample.shape[1:]))])
    hash_vals = np.dot(sample_flattened, position_weights)

    c = Counter(hash_vals)
    p = np.array(list(c.values()))
    p = p/p.sum()
    return np.sum( -p*np.log(p+1e-31) ) 


def entropy_estimator_same_cardinality(sample, flow, base):
    """ Estimates entropy using MC.

        Assumes that all dimensions have the same cardinality. """
    inv_sample = flow.reverse(sample) #inv_sample += 1e-7 
    entropy = tf.reduce_mean( -base.log_prob_ext(inv_sample)  ) 
    return entropy


class EntropyVaryingCardinalities:
    """ Estimates entropy using MC (handles dimensions with varying cardinalities).

        Sample with last dimension =max(cardinalities over variables) 
        needs to be wrapped down to the right number of categories for each variable/dimension.
        In consequence, several positions in the last dimension map to the same category.
        Probabilities estimates need to be adjusted accordingly.
    """

    def __init__(self, cardinalities, axis=-2):
        self.cardinalities = cardinalities
        self.axis = axis

    def __call__(self, sample, flow, base):
        entropy = entropy_estimator_same_cardinality(sample, flow, base)
        logger.info("upper bound entropy: %.2f" % entropy )

        # populate 1s to all equivalent positions
        sample = cardinality.unwrap_categories(sample, self.cardinalities, axis=self.axis)
        # estimate in the standard way
        entropy = entropy_estimator_same_cardinality(sample, flow, base)
        logger.info("unwrapped entropy: %.2f" % entropy )

        return entropy


class VariationalInference:
    """ Training discrete flows using variational inference. """
    
    def __init__(self, base, flow, log_prob=None,
                 temperature_annealing=TemperatureAnnealingExp(),
                 nsamples=100, max_niter=10000, noimprov_niter=100, min_niter=0,
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.01), 
                 entropy_estimator=entropy_estimator_same_cardinality, **kwargs):
        """
            Args:
                base  Base sampling distribution (an instance of CategoricalMixture).
                flow  A flow or mixture of flows transforming samples 
                      from the base distribution.
                log_prob  A function that takes a tensor nsamples x dimensionality
                          and returns a tensor of length = nsamples.
                          Used to estimate ELBO=log_prob+entropy of latent discrete variables.
                          It should output logs of model joint probabilities added to
                          entropies of non-discrete variables.
        """
        logger.debug("[VariationalInference] ignored kwargs=%s" % kwargs)

        self.log_prob = log_prob
        self.base = base
        self.flow = flow
        self.temperature_annealing = temperature_annealing

        self.nsamples = nsamples
        self.max_niter = max_niter
        self.min_niter = min_niter
        self.optimizer = optimizer
        self.noimprov_niter = noimprov_niter

        # progress recording
        self.best_base = None
        self.best_flow = None
        self.best_loss = float("inf")
        self.time_forward = 0.
        self.time_backward = 0.

        self.entropy_estimator = entropy_estimator

    @property
    def dtype(self):
        assert self.flow.dtype==self.base.dtype
        return self.flow.dtype

    @property
    def temperature(self):
        return self.flow.temperature

    @property
    def B(self):
        return self.flow.B

    @property
    def trainable_variables(self):
        return self.flow.trainable_variables

    def apply_gradients(self, tape, loss, iteration=None):
        grads = tape.gradient(loss, self.trainable_variables)   
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))     
        #for j, grad in enumerate(grads):
        #    assert tf.reduce_any( tf.math.is_nan(grad) )==False, \
        #        "grads[%i] = %s" % (j, grad)   

    def elbo(self, nsamples=None):
        if nsamples is None: nsamples = self.nsamples
        base_sample, mask = self.base.sample_extm(nsamples)
        sample = self.flow(base_sample, mask)
            
        #inv_sample = self.flow.reverse(sample) #inv_sample += 1e-7 #!
        #entropy = tf.reduce_mean( -self.base.log_prob_ext(inv_sample)  ) 
        entropy = self.entropy_estimator(sample, self.flow, self.base)
                    
        log_joint_prob = tf.reduce_mean( self.log_prob(sample) )
        return log_joint_prob + entropy     

    def fit(self, callback=lambda obj, iteration, loss: \
                           logger.info("%i. loss=%.2f" % (iteration, loss))):
        if self.log_prob is None:
            raise ValueError("[VariationalInference.fit] log_prob must be set!")

        last_improvement = 0
        for iteration in range(self.max_niter):
        
            self.flow.temperature = self.temperature_annealing(iteration)

            start_op_time = time.time()            
            with tf.GradientTape() as tape: loss = -self.elbo()       
            self.time_forward += time.time()-start_op_time 

            improved = loss+IMPROV_EPS < self.best_loss
            #if improved: # reevaluate on more samples to reduce sampling error
            #    new_loss = -self.elbo(self.nsamples*10)
            #    improved = loss+IMPROV_EPS < self.best_loss
            if improved: # Track the best solution
                try:
                    self.best_base = copy.deepcopy(self.base)
                    self.best_flow = copy.deepcopy(self.flow) if self.flow!=self.base else self.best_base
                except Exception as exc:
                    if self.best_flow is None:
                        logger.error("[ERROR][%s] Failed to make a copy of base & flow objects: %s" % (self, exc))
                    self.best_base = self.base
                    self.best_flow = self.flow
                    

                self.best_loss = loss.numpy() #new_loss.numpy()        
                last_improvement = iteration       

            # must be ivoked before approximation update
            callback(self, iteration, loss.numpy())

            start_op_time = time.time()            
            self.apply_gradients(tape, loss, iteration)
            self.time_backward += time.time()-start_op_time 
        
            if iteration>=self.min_niter and self.noimprov_niter<iteration-last_improvement: 
                logger.info("[VariationalInference.fit] No improvement in recent %i iterations. Stop." %\
                             self.noimprov_niter)
                break

        return iteration


class IterativeVariationalInference(VariationalInference):
    """ Trains iteratively flows keeping previously trained fixed. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainable_variables = None
        self._C = None

    @property
    def trainable_variables(self):
        #logger.debug("[IterativeVariationalInference] self._C=%i" % self._C)
        return self.flow.select_trainable_variables(self._C, self._C+1) 

    @property
    def C(self):
        """ Number of a component currently being trained. """
        return self._C

    def get_weights(self):
        # set mixing probs to equal split between all componets 0..C
        mixing_probs = np.zeros(self.B)
        mixing_probs[ : (self._C+1)] = 1.
        mixing_probs = mixing_probs / np.sum(mixing_probs) 
        rho = tf.constant(mixing_probs, dtype=self.base.mixing_probs.dtype)
        return rho

    def fit(self, callback=lambda obj, iteration, loss: \
                           logger.info("%i. loss=%.2f" % (iteration, loss))):
        total_niter = 0
        callback_proxy = lambda obj, i, loss: callback(obj, total_niter+i, loss)
        for C in range(0, self.B):
            self._C = C
            logger.info("[IterativeVariationalInference] Fitting component C=%i" % C)
            self.base.mixing_probs = self.get_weights()
            total_niter += super().fit(callback=callback_proxy)
        return total_niter


# Avoid infinities when calculating log-prob of a BVI(F) mixture
ZERO_WEIGHT = 1e-12


class BoostingVariationalInference(VariationalInference):
    """ Variational Boosting: Iteratively Refining Posterior Approximations. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainable_variables = None
        self._C = None
        self._rho_C_unconstrained = None
        self.base_prev_C = None
        self.flow_prev_C = None

    @property
    def C(self):
        """ Number of a component currently being trained. """
        return self._C

    @property
    def trainable_variables(self):
        vs = self.flow.select_trainable_variables(self._C, self._C+1)+[self._rho_C_unconstrained]
        #logger.debug("[BVI] self._C=%s => variables=%s" % (self._C, [v.name for v in vs]))
        return vs

    def get_weights(self):        
        prev_rho = [] if self._C==0 else self.base_prev_C.mixing_probs.numpy()[ : (self._C)]

        rho_C = tf.sigmoid(self._rho_C_unconstrained)            
        rho = [(1.-rho_C)*r for r in prev_rho] + [rho_C] + [ZERO_WEIGHT]*(self.B-self._C-1)
        rho = tf.stack(rho)
        #rho = rho/tf.reduce_sum(rho) #renormalize        
        #logger.debug("[BVI.get_weights] prev_rho=%s rho=%s" % (np.round(prev_rho, 3), np.round(rho, 3)))
        return rho

    def elbo(self):
        rho = self.get_weights()
        self.base.mixing_probs = rho

        # elbo1 of an existing approximation
        if self._C==0: 
            elbo1 = 0.0
        else:
            ## sample from old (q_C) approximation                                
            base_sample, mask = self.base_prev_C.sample_extm(self.nsamples)
            sample = self.flow_prev_C(base_sample, mask)
                
            # evaluate entropy using new (q_C+1) approximation
            #entropy = tf.reduce_mean( -self.base.log_prob_ext(self.flow.reverse(sample)) )
            entropy = self.entropy_estimator(sample, self.flow, self.base)
                
            log_joint_prob = tf.reduce_mean( self.log_prob(sample) )
            elbo1 = log_joint_prob + entropy

        # elbo2 of a new component
        base_sample, mask = self.base.sample_extm(self.nsamples) # sample from new (q_C+1) approximation
        sample = self.flow(base_sample, mask)
        #entropy = tf.reduce_mean( -self.base.log_prob_ext(self.flow.reverse(sample))  )             
        entropy = self.entropy_estimator(sample, self.flow, self.base)
            
        log_joint_prob = tf.reduce_mean( self.log_prob(sample) )
        elbo2 = log_joint_prob + entropy                               

        rho_C = rho[self._C]
        #logger.debug("[BVI.elbo] rho_C%s=%.2f elbo1=%.2f elbo2=%.2f" % (self._C, rho_C, elbo1, elbo2))
        return (1.-rho_C)*elbo1 + rho_C*elbo2 

    def _extend_mixture(self):  
        self.base = copy.deepcopy(self.best_base) # start from the best
        self.flow = copy.deepcopy(self.best_flow) if self.best_flow!=self.best_base else self.base
            
        # store prev approximation (q_C)
        #self.base_prev_C = copy.deepcopy(self.base)
        #self.flow_prev_C = copy.deepcopy(self.flow) if self.flow!=self.base else self.base_prev_C   
        self.base_prev_C = copy.deepcopy(self.best_base) # start from the best
        self.flow_prev_C = copy.deepcopy(self.best_flow) if self.best_flow!=self.best_base else self.base_prev_C 
 
    def fit(self, callback=lambda obj, iteration, loss: \
                           logger.info("%i. loss=%.2f" % (iteration, loss))):

        total_niter = 0
        # assures that iterations continue when components switch (due to +total_niter)
        callback_proxy = lambda obj, i, loss: callback(obj, total_niter+i, loss) 
        for C in range(0, self.B):
            logger.info("[IterativeVariationalInference] Fitting C=%i" % C)
    
            # initialize next weight
            self._rho_C_unconstrained = tf.Variable(np.random.random() if C!=0 else 100.0, 
                                                    name="rho_C%i_unconstrained" % C) 

            self._C = C
            total_niter += super().fit(callback=callback_proxy)
            self._extend_mixture()

        return total_niter
 

class BoostingVariationalInferenceAltering(BoostingVariationalInference): 
    """ Variational Boosting: alternating between training flows and weights. """

    def __init__(self, *args, **kwargs):
        """
            Args:
                switch_niter: switch between training flows and weights every niters.                
        """
        self.switch_niter = kwargs.pop("switch_niter", 25)
        super().__init__(*args, **kwargs)

    def apply_gradients(self, tape, loss, iteration):
        if (iteration // self.switch_niter) % 2 == 0:
            #logger.debug("[BVIAltering.apply_gradients] i=%i -> flows" % iteration)
            trainable_variables = self.flow.select_trainable_variables(self._C, self._C + 1)
        else:
            #logger.debug("[BVIAltering.apply_gradients] i=%i -> weights" % iteration)
            trainable_variables = [self._rho_C_unconstrained]
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))


class BoostingVariationalInferenceAlteringIndep(BoostingVariationalInference): 
    """ Variational Boosting: alternating between training flows and weights. """

    def __init__(self, *args, **kwargs):
        """
            Args:
                switch_niter: switch between training flows and weights every niters.  
                optimizer_weights: use a separate optimizer for weights than for flows.              
        """
        self.optimizer_weights = kwargs.pop("optimizer_weights", tf.keras.optimizers.RMSprop(lr=0.01))
        self.switch_niter = kwargs.pop("switch_niter", 25)
        super().__init__(*args, **kwargs)

    def apply_gradients(self, tape, loss, iteration):
        if (iteration // self.switch_niter) % 2 == 0:
            #logger.debug("[BVIAlteringIndep.apply_gradients] i=%i -> flows" % iteration)
            trainable_variables = self.flow.select_trainable_variables(self._C, self._C + 1)
            grads = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(grads, trainable_variables))
        else:
            #logger.debug("[BVIAlteringIndep.apply_gradients] i=%i -> weights" % iteration)
            trainable_variables = [self._rho_C_unconstrained]
            grads = tape.gradient(loss, trainable_variables)
            self.optimizer_weights.apply_gradients(zip(grads, trainable_variables))  


 

