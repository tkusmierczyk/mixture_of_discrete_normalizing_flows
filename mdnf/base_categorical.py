# -*- coding: utf-8 -*-
""" Factorized categorical (base) distribution. """


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import logging
logger = logging.getLogger(__name__)

from time_profiling import timing


class MultivariateFactorizedCategorical(tfd.OneHotCategorical):
    """ An improved wrapper of OneHotCategorical.
    
        Fixes a problem of OneHotCategorical with p=[0...1...0] => log_prob=NaN. 
    """
  
    def __init__(self, **kwargs):
        logger.debug("[MultivariateFactorizedCategorical] creating with: %s" % kwargs)

        kwargs.pop('mixing_probs', None) # ignore parameter
        kwargs.setdefault("dtype", tf.keras.backend.floatx())
        tfd.OneHotCategorical.__init__(self, **kwargs)
        assert len(self.probs.shape)==1 or len(self.probs.shape)==2

        v = self._logits if self._logits is not None else self._probs
        self._N = 1 if len(v.shape)==1 else v.shape[-2]
        self._K = v.shape[-1]

    def sample(self, n=1):
        return tfd.OneHotCategorical.sample(self, n)

    @staticmethod  
    def log_prob_independent_static(x_one_hot, probs):
        """ Assumes dimensions to be independent and does not sum them. """
        log_prob = []
        for d in range(x_one_hot.shape[-2]): #TODO vectorize
            x1 = x_one_hot[..., d, :]
            probs_x1 = tf.reduce_sum( tf.multiply(x1, probs[d]), 1) # sum over categories
            log_prob.append( tf.math.log(probs_x1) )
        log_prob = tf.stack(log_prob, -1)
        return log_prob          

    def log_prob(self, x_one_hot):
        """ Returns log-probability of a one-hot-encoded sample. 

            Assumes dimensions to be independent and does not sum them. 
            For empty inputs (=only zeros) returns log_prob = -inf.
        """
        assert x_one_hot.shape[-1]==self.K
        assert x_one_hot.shape[-2]==self.N
        return self.log_prob_independent_static(x_one_hot, self.probs)    
    
    def prob(self, x_one_hot):
        return tf.exp(self.log_prob(x_one_hot))

    def get_joint_probability_array(self):
        """ Returns N-dimensional array representing the final (joint) distribution. 

            Implemented primarly for debugging purposes.
        """
        import prob_recovery

        return prob_recovery.outer_product(self.probs)

    @property
    def probs(self):
        return tf.nn.softmax(self._logits) if self._logits is not None else self._probs

    @probs.setter
    def probs(self, v):
        prev = self._logits if self._logits is not None else self._probs
        assert prev.shape[-1]==v.shape[-1], "%s=prev.shape[-1]!=v.shape[-1]=%s" % (prev.shape[-1], v.shape[-1])
        self._probs = v #!
        self._logits = None

    @property
    def logits(self):
        return self._logits    

    @logits.setter
    def logits(self, v):
        prev = self._logits if self._logits is not None else self._probs
        assert prev.shape==v.shape
        #self._logits.assign(v)
        self._logits = v #!
        self._probs = None

    @property
    def N(self): 
        return self._N
        
    @property
    def K(self):    
        return self._K






