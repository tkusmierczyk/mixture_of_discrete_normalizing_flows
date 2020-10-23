# -*- coding: utf-8 -*-
""" Individual discrete flows. """


import numpy as np
import tensorflow as tf  

import logging
logger = logging.getLogger(__name__)

try:
    import edward2 as ed
    made = ed.layers
except:
    logger.warning(""" WARNING: Failed to import Edward2! Certain flow types may be not available! """)

import flows_edward2 as fed
import flows_edward2_made as made
from flows_factorized import *

from flows_transformations import build_rnn_model, CopiableMADE, CopiableMADELocScale


class DiscreteFlow(tf.keras.layers.Layer):     
    """ Individual discrete flow (possibly with several layers). """

    def __init__(self, N=None, K=None, temperature=1.0, layers=[("M", [128])],
                 **kwargs):
        """
            Args:
                layers  Either specification of layers in form of pairs 
                        (layer_type, params) or simply a list of layers. 
                temperature  Initial temperature for Straight-Through estimators.
        """
        super(DiscreteFlow, self).__init__(**kwargs)
        self._N = N
        self._K = K

        self._temperature = temperature
        self._layers = layers
        
        if N is not None and K is not None:
            self.build([1,N,K]) # we build immediately to have access to temperature

    def build(self, input_shape):
        N, K = input_shape[-2: ]
        assert self._N is None or self._N==N
        assert self._K is None or self._K==K
        self._N = N
        self._K = K

        layers = parse_layers_specification(self._layers, N, K, 
                                            self._temperature, 
                                            dtype=self.dtype) 
        self.sequential = tf.keras.Sequential(layers) 
        super().build(input_shape)    
        logger.debug("[DiscreteFlow.build] name=%s layers=%s" % 
                        (self.name, self.sequential.layers))      
                       
    def __str__(self):
        return ("DiscreteFlow(N=%s (#dims), K=%s (#categories) " % (self.N, self.K))

    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K       

    def call(self, x):
        #for f in self.sequential.layers: 
        #    x = f(x)
        x = self.sequential(x)
        return x
     
    def reverse(self, x):
        for f in self.sequential.layers[-1::-1]:
            x = f.reverse(x)
        return x

    def select_trainable_variables(self, *args, **kwargs):
        return self.trainable_variables

    @property
    def temperature(self):
        t = None
        for f in self.sequential.layers:
            if hasattr(f, 'temperature'):
                if t is None: t = f.temperature
                elif t != f.temperature:
                    raise ValueError("No unique temperature value set for all flows (%s!=%s)!" % \
                                     (t, f.temperature))
        #if tf.is_tensor(t): t = t.numpy()
        return t          

    def set_temperature(self, t):
        for f in self.sequential.layers:
            if hasattr(f, 'set_temperature'): #handles objects having set_temperature method
                f.set_temperature(t)
            elif hasattr(f, 'temperature'): #handles edward2 layers where temperature is a Variable
                f.temperature.assign(t)        

    @temperature.setter
    def temperature(self, t):
        self.set_temperature(t)


class VariablesShuffling(tf.keras.layers.Layer):
    """ Wrapper that shuffles a flow input and shuffles back its output."""

    def __init__(self, flow, N=None, K=None, **kwargs):
        super().__init__(**kwargs)
        self.flow = flow
        self.shuffling = None
        if N is not None and K is not None: self.build([N, K])

    def build(self, input_shape):
        if self.shuffling is not None: return

        N = input_shape[-2]

        shuffling = np.arange(N)
        np.random.shuffle(shuffling)
        
        inverted_shuffling = np.arange(N)
        inverted_shuffling[shuffling] = np.arange(N)

        self.shuffling = shuffling
        self.inverted_shuffling = inverted_shuffling
        super().build(input_shape)  
        logger.debug("[VariablesShuffling.build] name=%s shuffling=%s..." % \
                    (self.name, str(self.shuffling[:100])[:200]))

    def call(self, x):    
        N = x.shape[-2]
        assert len(self.shuffling)==N
        
        x = tf.gather(x, self.shuffling, axis=-2)
        x = self.flow(x)
        x = tf.gather(x, self.inverted_shuffling, axis=-2)
        return x

    def reverse(self, x):    
        x = tf.gather(x, self.shuffling, axis=-2)
        x = self.flow.reverse(x)
        x = tf.gather(x, self.inverted_shuffling, axis=-2)
        return x


class VariablesReverse(tf.keras.layers.Layer):
        
    def __init__(self, **kwargs):
        super(VariablesReverse, self).__init__(**kwargs)
    
    def call(self, x, **kwargs):
        return tf.reverse(x,[-2])    
    
    def reverse(self, x):
        return tf.reverse(x,[-2])


class DummyFlow(tf.keras.layers.Layer):
    """ Identity mapping. """
        
    def __init__(self, temperature=None, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def call(self, x, **kwargs):
        return x    
    
    def reverse(self, x):
        return x



def parse_layers_specification(layers_specification, N, K, temperature, 
                               dtype=tf.keras.backend.floatx()):
    """ Constructs flows according to specification.

        Args:
            layers_specification  A list consisting either of flow objects or 
             tuples (flow_type_string, parameters) describing flows to be created.
             Check the code to see supported types.
            N  Input dimensionality
            K  Input cardinality
            temperature  ST estimator parameter
    """

    # the temperature will be traced by #@tf.function 
    # only if is an instance of tf.Variable
    # but we also don't want to train it
    if not tf.is_tensor(temperature):            
        temperature = tf.Variable(temperature, trainable=False, dtype=dtype)

    layers = []
    for j, layer_specification in enumerate(layers_specification):     
      
        if callable(layer_specification): 
            layer = layer_specification
        else:

            try: layer_type, args = layer_specification
            except: raise ValueError("Cannot parse layer specification = %s!" % layer_specification)

            if isinstance(args, int) or isinstance(args, float): args = [args]

            if layer_type=="MR" or layer_type.lower()=="made":  # loc flow  
                logger.debug("[flows.parse_layers_specs] creating MADE with randomized hidden units.")
                layer = fed.DiscreteAutoregressiveFlow(
                            CopiableMADE(K, hidden_dims=args, hidden_order="random"), temperature) 

            elif layer_type=="M" or layer_type=="MO" or layer_type.lower()=="made_ordered":  # loc flow  
                logger.debug("[flows.parse_layers_specs] creating MADE with ordered hidden units.")
                layer = fed.DiscreteAutoregressiveFlow(
                            CopiableMADE(K, hidden_dims=args, hidden_order="left-to-right"), temperature) 

            elif layer_type=="MEd": # Edward2 layers
                logger.debug("[flows.parse_layers_specs] creating Edward2 MADE with loc (hidden_dims=%s)." % args)
                network_ = ed.layers.MADE(K, hidden_dims=args, hidden_order="left-to-right") 
                layer = ed.layers.DiscreteAutoregressiveFlow(network_, temperature)

            elif layer_type=="M2": # loc+scale flow
                logger.debug("""[flows.parse_layers_specs] creating MADE (Edward2) with loc and scale 
                                (hidden_dims=%s).""" % args)
                made_class = CopiableMADELocScale
                network = made_class(K*2, hidden_dims=args, hidden_order="left-to-right") 
                layer = ed.layers.DiscreteAutoregressiveFlow(network, temperature)

            elif layer_type=="PM" or layer_type.lower()=="made_partial":
                categories, hidden_dims = args
                logger.debug("[flows.parse_layers_specs] creating partial MADE with categories=%s hidden=%s" % args)
                made = CopiableMADE(len(categories), hidden_dims=hidden_dims, hidden_order="left-to-right")
                layer = fed.DiscreteAutoregressivePartialFlow(K, categories, made, temperature)
            
            elif layer_type=="L" or  layer_type.lower()=="lstm":
                logger.debug("[flows.parse_layers_specs] creating LSTM with %i units." % args[0])
                rnn = build_rnn_model(K, rnn_units=args[0], embedding_dim=0, rnn_type=tf.keras.layers.LSTM)
                layer = fed.DiscreteAutoregressiveFlow(rnn, temperature)

            elif layer_type=="G" or  layer_type.lower()=="gru":
                logger.debug("[flows.parse_layers_specs] creating GRU with %i units." % args[0])
                rnn = build_rnn_model(K, rnn_units=args[0], embedding_dim=0, rnn_type=tf.keras.layers.GRU)
                layer = fed.DiscreteAutoregressiveFlow(rnn, temperature)

            elif layer_type=="LE" or  layer_type.lower()=="lstm_embedding":
                logger.debug("[flows.parse_layers_specs] creating LSTM with %i units & embedding." % args[0])
                rnn = build_rnn_model(K, rnn_units=args[0], embedding_dim=max(K//4, 2),
                                       rnn_type=tf.keras.layers.LSTM)
                layer = fed.DiscreteAutoregressiveFlow(rnn, temperature)

            elif layer_type=="GE" or  layer_type.lower()=="gru_embedding":
                logger.debug("[flows.parse_layers_specs] creating GRU with %i units & embedding." % args[0])
                rnn = build_rnn_model(K, rnn_units=args[0], embedding_dim=max(K//4, 2), 
                                      rnn_type=tf.keras.layers.GRU)
                layer = fed.DiscreteAutoregressiveFlow(rnn, temperature)

            elif layer_type=="m":  # Bipartite loc flow  
                mask = tf.constant(np.random.choice([0,1], N), dtype='int32')
                #layer = ed.layers.DiscreteBipartiteFlow(
                #            CopiableMADE(K, hidden_dims=args), mask=mask, temperature=temperature)
                layer = fed.DiscreteBipartiteFlow(
                            CopiableMADE(K, hidden_dims=args), mask=mask, temperature=temperature)  

            elif layer_type=="S0":
                shuffling = np.arange(K)
                np.random.shuffle(shuffling)
                shuffling = (shuffling+j*21)%K
                assert len(shuffling)==len(set(shuffling))
                layer = DiscreteFlowSubset(N, K, args[0], layers=args[1], temperature=temperature)          

            elif layer_type=="S":
                layer = DiscreteFlowSubsetIndependentDims(N, K, args[0], layers=args[1],
                                                          temperature=temperature)          

            elif layer_type=="R":  
                layer = VariablesReverse()
    
            elif layer_type=="F":
                logger.debug("[flows.parse_layers_specs] creating Factorized Flow with random init.")
                logits = tf.Variable( tf.random.normal((N,K), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlow(N, K, logits=logits,
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="FU":
                logger.debug("[flows.parse_layers_specs] creating Factorized Flow with uniform init.")
                logits = tf.Variable( tf.ones((N,K), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlow(N, K, logits=logits, 
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="F2":
                logger.debug("[flows.parse_layers_specs] creating DiscreteFactorizedFlowLocScale with random init.")
                logits = tf.Variable( tf.random.normal((N,K), dtype=dtype), dtype=dtype)
                logits_scale = tf.Variable( tf.random.normal((N,K), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlowLocScale(N, K, logits=logits, logits_scale=logits_scale,
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="F2U":
                logger.debug("[flows.parse_layers_specs] creating DiscreteFactorizedFlowLocScale with uniform init.")
                logits = tf.Variable( tf.ones((N,K), dtype=dtype), dtype=dtype)
                logits_scale = tf.Variable( tf.ones((N,K), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlowLocScale(N, K, logits=logits, logits_scale=logits_scale,
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="PF" or layer_type.lower()=="factorized_partial":
                categories = args
                logger.debug("""[flows.parse_layers_specs] creating partial Factorized Flow with 
                                categories=%s (random init) N=%s K=%s.""" % (categories, N,K))
                logits = tf.Variable( tf.random.normal((N, len(categories)), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlowPartial(N, K, categories, logits=logits, 
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="PFU" or layer_type.lower()=="factorized_uniform_partial":
                categories = args
                logger.debug("""[flows.parse_layers_specs] creating partial Factorized Flow with 
                                categories=%s (uniform init)""" % categories)
                logits = tf.Variable( tf.ones((N, len(categories)), dtype=dtype), dtype=dtype)
                layer = DiscreteFactorizedFlowPartial(N, K, categories, logits=logits, 
                                               temperature=temperature, dtype=dtype)

            elif layer_type=="I":  
                layer = DummyFlow()

            else: raise NameError("Unknown layer type: %s" % layer_type)

        layers.append(layer)
    return layers 




