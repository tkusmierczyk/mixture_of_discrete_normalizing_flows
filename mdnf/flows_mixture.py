# -*- coding: utf-8 -*-
""" Mixture of discrete normalizing flows. """


import numpy as np
import tensorflow as tf

from flows import DiscreteFlow, parse_layers_specification
from time_profiling import timing

import logging
logger = logging.getLogger(__name__)



class DiscreteFlowsMixture(tf.keras.layers.Layer):
    """ Mixture of B discrete normalizing flows, every with N variables of K categories. """
    
    def __init__(self, N, K, B, flows=None, 
                 temperature=None, 
                 single_flow_layers=[ ("M", [64, 64]) ],
                 components_specification=None, 
                 **kwargs): 
        """
            Args:
                flows  A list of components
                temperature  ST hyperparameter
                single_flow_layers  Specification of layers. 
                    If used all components will follow the same design.

            Either a list of flows or a temperature for newly created 
            (according to single_flow_layers) flows should be provided.
        """
        super(DiscreteFlowsMixture, self).__init__(**kwargs)

        self._N = N
        self._K = K
        self._B = B      
        assert temperature is not None or flows is not None, \
            "You have to provide either a list of flows or a temperature for newly created flows!"

        if flows is None: 
            if components_specification is not None:
                assert len(components_specification)==B, \
                        "components_specification should contain one layer specification per flow"
                flows = parse_layers_specification(components_specification, N, K, temperature, dtype=self.dtype) 
            elif single_flow_layers is not None:
                flows = [DiscreteFlow(N, K, temperature, layers=single_flow_layers) 
                         for _ in range(B)]
            else:
                raise ValueError("You have to fix one of the following:" + \
                                 "flows, components_specification or single_flow_layers!")
        assert len(flows)==B
        self.flows = flows

    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K 

    @property
    def B(self):
        return self._B 

    @timing
    def call(self, x, mask=None, **kwargs):
        N, K, B = self.N, self.K, self.B
        assert x.shape[-1]==K, "x.shape[-1]=%s K=%s" % (x.shape[-1], K)
        assert x.shape[-2]==B, "x.shape[-2]=%s B=%s" % (x.shape[-2], B)
        assert x.shape[-3]==N, "x.shape[-3]=%s N=%s" % (x.shape[-3], N)       
        assert mask is None or len(x.shape)==len(mask.shape)

        # list of B elements of size n x N x K
        blocks = [flow(x[..., b, :]) for b, flow in enumerate(self.flows)] 
        x = tf.stack(blocks, axis=-2)
        if mask is not None: 
            x *= mask
        x = tf.reduce_sum(x, axis=-2)                
        return x  
      
    @timing
    def reverse(self, x):
        N, K, B = self.N, self.K, self.B
        assert x.shape[-1]==K, "reversed sample should have K=%s but has=%s" % (K, x.shape[-1])
        assert x.shape[-2]==N, "reversed sample should have N=%s but has=%s" % (N, x.shape[-2])

        blocks = [flow.reverse(x) for flow in self.flows]
        x = tf.stack(blocks, -2)
        return x

    def select_trainable_variables(self, first_flow_index=0, end_flow_index=None):
        trainable_vars = None
        for i in np.arange(len(self.flows))[first_flow_index: end_flow_index]:
            if trainable_vars is None: 
                trainable_vars = self.flows[i].trainable_variables
            else:                      
                trainable_vars += self.flows[i].trainable_variables
        return trainable_vars

    @property
    def temperature(self):
        t = None
        for f in self.flows:
            if hasattr(f, 'temperature'):
                if t is None: t = f.temperature
                elif t != f.temperature:
                    raise ValueError("No unique temperature value set for all flows!")
        if tf.is_tensor(t): t = t.numpy()
        return t     

    def set_temperature(self, t):
        for f in self.flows:
            f.temperature = t

    @temperature.setter
    def temperature(self, t):
        self.set_temperature(t)

