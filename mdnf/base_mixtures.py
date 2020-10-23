# -*- coding: utf-8 -*-
""" Mixture of categorical distributions. """


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import logging
logger = logging.getLogger(__name__)

from time_profiling import timing


class CategoricalMixture(tfd.Distribution):
    """ Mixture of categorical distributions. 

        Basic naive (reference) implementation;
        for practical purposes use FactorizedCategoricalMixture.
    """

    def __init__(self, mixing_distribution, components, **kwargs):
        """
        Args:
            mixing_distribution  OneHotCategorical distribution choosing one from components.
            components  A list of categorical distributions (not necessarily factorized).
        """

        kwargs.setdefault('dtype', tf.keras.backend.floatx())
        kwargs.setdefault('reparameterization_type', tfd.NOT_REPARAMETERIZED)
        kwargs.setdefault('validate_args', False)
        kwargs.setdefault('allow_nan_stats', True)
        super().__init__(**kwargs)

        assert len(mixing_distribution.batch_shape) == 0
        assert len(mixing_distribution.event_shape) == 1
        assert mixing_distribution.event_shape[-1] == len(components)
        # let's check at least the first one 
        # (assuming the rest follows with the same shapes)
        assert len(components[0].event_shape) == 1 
        assert len(components[0].batch_shape) >= 1

        self._distributions = components
        self._mixing_distribution = mixing_distribution

        self._N = components[0].batch_shape[-1]
        self._K = components[0].event_shape[-1]
        self._B = len(components)
        
        logger.debug(self.__str__())

    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K

    @property
    def B(self):
        return self._B

    def _batch_shape(self):
        return np.array([])

    def _event_shape(self):
        return np.array([self.N, self.K])

    def __str__(self):
        return ("%s(N=%s (#dims), B=%s (#components), K=%s (#categories) " + \
                "mixing_probs=%s...)") % (self.__class__, self.N, self.B, self.K, \
                                          str(self.mixing_probs)[:100])

    @staticmethod
    def log_prob_ext_static(x_ext, mixing_probs, distributions): 
        log_mixing_probs = tf.math.log(mixing_probs)
       
        B, K = x_ext.shape[-2: ]
        log_probs = []
        for b in range(B):
            x_one_hot = x_ext[..., b, :]
            log_prob = distributions[b].log_prob(x_one_hot) 
            log_prob = tf.reduce_sum(log_prob, -1) 
            log_probs.append( log_prob+log_mixing_probs[b] )        
        return tf.math.reduce_logsumexp(tf.stack(log_probs, 0), 0) 

    @timing
    def log_prob_ext(self, x_ext):
        """ Returns log-probabilities of (extended) samples of size (batch x N x B x K). 
                        
            Can handle samples having 1 for more than one component.
            In such case, probabilities for respective components are added.
            Operates on log-scale and uses logsumexp.
            See: sample_ext.
        """
        assert x_ext.shape[-1]==self.K
        assert x_ext.shape[-2]==self.B
        assert x_ext.shape[-3]==self.N
        return self.log_prob_ext_static(x_ext, self.mixing_probs, self.distributions)
    
    def prob_ext(self, x_ext):
        return tf.math.exp(self.log_prob_ext(x_ext))       

    @timing        
    def sample_ext(self, n=1):
        """ Returns n samples of size NxBxK from the mixture. 
        
            Samples from each of the components are stacked 
            along the before-last dimension:
            zeros everywhere apart from the cells matching selected components.
            Optimized for large n and small B.            
        """        
        component_samples, mask = self.sample_extm(n)
        return component_samples*mask

    @timing        
    def sample_extm(self, n=1):
        # sampling from all components
        component_samples = [self._distributions[i].sample(n) for i in range(self.B)]
        component_samples = tf.stack(component_samples, axis=-2) # n x N x B x K
    
        selected_components = self._mixing_distribution.sample(n) # n x B
        mask = tf.transpose(selected_components) # B x n 
        mask = tf.broadcast_to(mask, (self.N, self.K, self.B, n))
        mask = tf.transpose(mask, perm=[3,0,2,1] ) # n x N x B x K
        return component_samples, mask

    def get_joint_probability_array(self):
        """ Returns N-dimensional array representing the final (joint) distribution of the mixture. 
        
            Implemented primarly for debugging purposes.
        """
        import prob_recovery

        prob_array = 0.
        mixing_probs = self.mixing_probs
        if isinstance(mixing_probs, tf.Variable): 
            mixing_probs = mixing_probs.numpy()
        for p, d in zip(mixing_probs, self._distributions):
            prob_array += p*prob_recovery.outer_product(d.probs)
        return prob_array.numpy()           

    @property
    def mixing_probs(self):
        return self._mixing_distribution.probs

    @mixing_probs.setter
    def mixing_probs(self, v):
        self._mixing_distribution.probs = v

    def _set_mixing_logits(self, value):
        self._mixing_distribution.logits = value
    mixing_logits = property(fset=_set_mixing_logits)  # only a setter
    del _set_mixing_logits  # optional: delete the unneeded setter function
                
    @property
    def distributions(self):
        return self._distributions

    @property
    def components(self):
        return self.distributions

    @property
    def probs(self):
        return [d.probs for d in self.distributions]
                
    @property
    def temperature(self):
        """ Retrieves temperature of the mixing distribution and component distributions. """
        t = self._mixing_distribution.temperature \
            if hasattr(self._mixing_distribution, 'temperature') else None
        for f in self.distributions:
            if hasattr(f, 'temperature'):
                if t is None: t = f.temperature
                elif t != f.temperature:
                    raise ValueError("No unique temperature value set for all distributions!")
        return t     
    
    @temperature.setter
    def temperature(self, t):
        """ Sets temperature for the mixing distribution and component distributions. """
        if hasattr(self._mixing_distribution, 'temperature'):
            self._mixing_distribution.temperature.assign(t)  
        for f in self.distributions:
            if hasattr(f, 'temperature'): 
                f.temperature.assign(t)

    @timing
    def sample(self, n=1):
        """ Employs sample_ext to obtain n samples from the mixture, each of size NxK. """
        samples_ext = self.sample_ext(n)
        return tf.reduce_sum(samples_ext, -2)   

    def log_prob(self, x_one_hot):
        """ Returns log-probabilities of a one-hot encoded batch of samples. 

            Operates on log-scale and uses logsumexp.
        """
        assert x_one_hot.shape[-1]==self.K
        assert x_one_hot.shape[-2]==self.N
        
        log_mixing_probs = tf.math.log(self.mixing_probs)
        log_probs = []
        for b in range(self.B):
            log_prob = self._distributions[b].log_prob(x_one_hot)
            log_prob = tf.reduce_sum(log_prob, -1)  
            log_probs.append( log_prob+log_mixing_probs[b] )        
        return tf.math.reduce_logsumexp(tf.stack(log_probs, 0), 0)   
                    
    def prob(self, x_one_hot):
        """ Returns probabilities of a one-hot encoded batch of samples. """
        return tf.math.exp(self.log_prob(x_one_hot))  
            

class FactorizedCategoricalMixture(CategoricalMixture):
    """ Mixture of factorized categorical distributions. 

        An efficient implementation exploiting factorization of components.
    """

    def __init__(self, mixing_distribution, components, **kwargs):
        super().__init__(mixing_distribution, components, **kwargs)

        self._components_probs = tf.constant( tf.stack([component.probs for component in self.components], -2) )
        self._components_distribution = tfd.OneHotCategorical(probs=self._components_probs, dtype=self.dtype)
        self.uniform_mask = False
        
    @staticmethod
    def log_prob_ext_static(x_ext, mixing_probs, component_probs): 
        log_mixing_probs = tf.math.log(mixing_probs) # B
        prob = tf.reduce_sum(component_probs*x_ext, -1) # sum over categories => n x N x B
        log_prob = tf.math.log(prob)
        log_prob = tf.reduce_sum(log_prob, -2) # sum over variables (dims) => n x B
        log_prob = log_prob + log_mixing_probs
        log_prob = tf.math.reduce_logsumexp(log_prob, -1) # sum over components => n
        return log_prob

    @timing
    def log_prob_ext(self, x_ext):
        """ Returns log-probabilities of (extended) samples of size (batch x N x B x K). 
                        
            Can handle samples having 1 for more than one component.
            In such case, probabilities for respective components are added.
        """
        assert x_ext.shape[-3:]==self._components_distribution.batch_shape+self._components_distribution.event_shape
        return self.log_prob_ext_static(x_ext, self.mixing_probs, self._components_probs)

    def _sample_mask(self, n):
        if self.uniform_mask: # if the flag is fixed (almost) deterministically split into equal parts
            return tf.one_hot((np.arange(n)+np.random.randint(self.B))%self.B, self.B, dtype=self._mixing_distribution.dtype)
        return self._mixing_distribution.sample(n)

    @timing 
    def sample_extm(self, n=1):     
        """ Returns n samples and a mask of size NxBxK from the mixture. """         
        component_samples = self._components_distribution.sample(n)

        mask = self._sample_mask(n)
        if len(mask.shape)==2: 
            mask = mask[..., None, :, None]
        elif len(mask.shape)==3: 
            mask = mask[..., None]
        else:
            raise ValueError("""Mixing distribution samples must either have 
                                shape nsamplesxB or nsamplesxNxB!""")

        return component_samples, mask

    def sample_ext(self, n=1):
        """ Returns n samples of size NxBxK from the mixture. """      
        component_samples, mask = self.sample_extm(n)
        return component_samples*mask


class IndependentCategoricalMixture(CategoricalMixture):
    """ Mixture of categorical distributions 
        with independence between variables (dim=-2) assumed. 
    """

    @staticmethod
    def log_prob_ext_independent_static(x_ext, mixing_probs, distributions):
        log_mixing_probs = tf.math.log(mixing_probs)

        B, K = x_ext.shape[-2:]
        log_probs = []
        for b in range(B):
            x_one_hot = x_ext[..., b, :]
            log_prob = distributions[b].log_prob(x_one_hot) 
            log_probs.append( log_prob+log_mixing_probs[b] )     
        log_probs = tf.stack(log_probs, 0) # B x batch x N
        log_probs = tf.math.reduce_logsumexp(log_probs, 0) # sum over B mixture components        
        return tf.reduce_sum(log_probs, -1) # sum over N

    @timing
    def log_prob_ext_independent(self, x_ext):
        """ Returns log-probabilities of samples of size (batch x N x B x K) 
            assuming independence between dimensions. 
        """
        assert x_ext.shape[-1]==self.K
        assert x_ext.shape[-2]==self.B
        assert x_ext.shape[-3]==self.N
        return self.log_prob_ext_independent_static(x_ext, self.mixing_probs, self.distributions)

    def log_prob_ext(self, x_ext):
        """ Delegates execution to log_prob_ext_independent. """
        return self.log_prob_ext_independent(x_ext)     


class FactorizedIndependentCategoricalMixture(FactorizedCategoricalMixture, IndependentCategoricalMixture):
    """ Mixture of categorical distributions with independence between variables (dim=-2) assumed.

        An efficient implementation exploiting factorization of components.
    """

    @staticmethod
    def log_prob_ext_independent_static(x_ext, mixing_probs, component_probs):
        log_mixing_probs = tf.math.log(mixing_probs)  # B or N x B
        prob = tf.reduce_sum(component_probs*x_ext, -1) # sum over categories => n x N x B
        log_prob = tf.math.log(prob)
        log_prob = log_prob + log_mixing_probs # n x N x B
        log_prob = tf.math.reduce_logsumexp(log_prob, -1) # sum over B mixture components  => n x N
        return tf.reduce_sum(log_prob, -1) # sum over N

    @timing
    def log_prob_ext_independent(self, x_ext):
        """ Returns log-probabilities of (extended) samples of size (batch x N x B x K). 
                        
            Can handle samples having 1 for more than one component.
            In such case, probabilities for respective components are added.
        """
        assert x_ext.shape[-3:]==self._components_distribution.batch_shape+self._components_distribution.event_shape
        result =  self.log_prob_ext_independent_static(x_ext, self.mixing_probs, self._components_probs)
        return result

    def log_prob_ext(self, x_ext):
        """ Delegates execution to log_prob_ext_independent. """
        return self.log_prob_ext_independent(x_ext)  



