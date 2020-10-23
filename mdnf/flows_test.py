# -*- coding: utf-8 -*-

import unittest

import flows

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


import logging
logger = logging.getLogger(__name__)

import copy


def forward(flow, samples):
    return flow(samples)


def backward(flow, samples):
    return flow.reverse(samples)


class TestFlows(unittest.TestCase):

    def __init__(self, *args):
        super(TestFlows, self).__init__(*args)
        #tf.keras.backend.set_floatx('float64')

        np.random.seed(13)
        tf.random.set_seed(13)
      
        N = 17
        K = 23 # must be prime to work with scale!=1
        base = tfp.distributions.OneHotCategorical(logits=np.random.randn(N,K))
        logger.debug("[init] base=%s" % (base,) )

        self.base_samples = tf.cast(base.sample(256), dtype='float32')
        logger.debug("[init] samples=%s (%s)" % (self.base_samples.shape, self.base_samples.dtype))

        self.empty_samples = tf.zeros((256,N,K), dtype=self.base_samples.dtype)
        logger.debug("[init] empty samples=%s (%s)" % (self.empty_samples.shape, self.empty_samples.dtype))

        s = np.zeros((100,N,K))
        s[:,:,13] = 1.
        self.equal_samples = tf.constant(s, dtype=self.base_samples.dtype)
        logger.debug("[init] equal samples=%s (%s)" % (self.equal_samples.shape, self.equal_samples.dtype))

        s = np.zeros((100,N,K))
        s[:,:,22] = 1.
        self.equal_samples2 = tf.constant(s, dtype=self.base_samples.dtype)
        logger.debug("[init] equal samples2=%s (%s)" % (self.equal_samples2.shape, self.equal_samples2.dtype))

        self.N = N
        self.K = K

        self.layers_specs = [ 
                    [("M",[12, 16])],
                    [("M",[N])],
                    [("M",[12, 16]), ("M",[16, 12])],
                    [("MEd", N)],
                    [("M2", N)],
                    [("F", None)], 
                    [("FU", None)], 
                    [("FU", None), ("F", None)], 
                    ]

    def setUp(self):
        np.random.seed(13)
        tf.random.set_seed(13)
   
    def test_discrete_flows_invertability(self):
        """Tests various flows by passing samples both directions and checking if input is recovered."""
        for i, layers in enumerate(self.layers_specs):
            logger.debug("#"*30+" [%i/%i] testing invertability of %s" % (i+1, len(self.layers_specs), layers))
            flow = flows.DiscreteFlow(self.N, self.K, temperature=5.0, layers=copy.deepcopy(layers))
            self.assert_forward_reverse(flow, str(layers))

    def test_discrete_flows_deterministic_forward(self):
        """Tests various flows by passing set of exactly the same samples and checking if result is the same."""
        N = self.N
        for i, layers in enumerate(self.layers_specs):
            logger.debug("#"*30+" [%i/%i] testing repeated samples (forward) %s" % \
                         (i+1, len(self.layers_specs), layers))
            flow = flows.DiscreteFlow(self.N, self.K, temperature=5.0, layers=copy.deepcopy(layers))

            out = flow(self.equal_samples)         
            self.assertTrue( sum([bool(tf.reduce_all(out[n] == out[n+1])) for n in range(N-1)])==N-1, 
                            "all samples the same")

            out = flow(self.equal_samples2)         
            self.assertTrue( sum([bool(tf.reduce_all(out[n] == out[n+1])) for n in range(N-1)])==N-1, 
                            "all samples the same")


    def test_discrete_flows_deterministic_reverse(self):
        """Tests various flows by passing set of exactly the same samples and checking if result is the same."""
        N = self.N
        for i, layers in enumerate(self.layers_specs):
            logger.debug("#"*30+" [%i/%i] testing repeated samples (reverse) %s" % \
                          (i+1, len(self.layers_specs), layers))
            flow = flows.DiscreteFlow(self.N, self.K, temperature=5.0, layers=copy.deepcopy(layers))

            out = flow.reverse(self.equal_samples)         
            self.assertTrue( sum([bool(tf.reduce_all(out[n] == out[n+1])) for n in range(N-1)])==N-1, 
                             "all samples the same")

            out = flow.reverse(self.equal_samples2)         
            self.assertTrue( sum([bool(tf.reduce_all(out[n] == out[n+1])) for n in range(N-1)])==N-1, 
                             "all samples the same")

    def test_discrete_flows_different_inputs(self):
        """Tests various flows by passing two different inputs and checking if outputs differ."""
        N = self.N
        for i, layers in enumerate(self.layers_specs):
            logger.debug("#"*30+" [%i/%i] testing repeated samples (reverse) %s" % \
                          (i+1, len(self.layers_specs), layers))
            flow = flows.DiscreteFlow(self.N, self.K, temperature=5.0, layers=copy.deepcopy(layers))

            out = flow(self.equal_samples[0:1])         
            out2 = flow(self.equal_samples2[0:1])         
            self.assertTrue(out.shape==out2.shape)
                        
            self.assertTrue( sum(bool(tf.reduce_any(out[0,n]!=out2[0,n])) for n in range(self.N))>0, \
                             "at least one dim in outputs must differ" )

    def test_temperature_update(self):
        """Test if temperature updates affect gradients."""
        flow = flows.DiscreteFlow(self.N, self.K, 
                temperature=10., layers=[("M",[256, 256]), ("F", None), ("M",[])]) 

        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(forward(flow, self.base_samples))
        g1 = tape.gradient(loss, flow.trainable_variables)

        with tf.GradientTape() as tape:
            flow.temperature = 1.0
            loss = -tf.reduce_sum(forward(flow, self.base_samples))
        g2 = tape.gradient(loss, flow.trainable_variables)

        with tf.GradientTape() as tape:
            flow.temperature = 0.1
            loss = -tf.reduce_sum(forward(flow, self.base_samples))
        g3 = tape.gradient(loss, flow.trainable_variables)

        difference_g1g2 = sum( np.sum(abs(e1-e2)) for e1, e2 in zip(g1,g2) )
        difference_g1g3 = sum( np.sum(abs(e1-e2)) for e1, e2 in zip(g1,g3) )

        logger.debug("Total gradient difference for temperature 10 vs 1: %s" % difference_g1g2)
        logger.debug("Total gradient difference for temperature 10 vs 0.1: %s" % difference_g1g3)

        self.assertNotEqual(difference_g1g2, 0.0)
        self.assertNotEqual(difference_g1g3, 0.0)
        self.assertNotEqual(difference_g1g3, difference_g1g2)


    def assert_forward_reverse(self, flow, msg="", precision=4):
        logger.debug("[assert_forward_reverse] flow = %s" % flow)

        z = forward(flow, self.base_samples)
        error = tf.reduce_max( tf.math.abs(backward(flow, z)-self.base_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(backward(flow, z)-self.base_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for forward-reverse %s: %s (%s/%s errors)" % \
                     (msg, error, num_errors, len(self.base_samples)) )
        self.assertAlmostEqual(error, 0., precision)        
        error1fr = error

        z = backward(flow, self.base_samples)
        error = tf.reduce_max( abs(forward(flow, z)-self.base_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(forward(flow, z)-self.base_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for reverse-forward %s: %s (%s/%s errors)" % \
                     (msg, error, num_errors, len(self.base_samples)))   
        self.assertAlmostEqual(error, 0., precision)  
        error1rf = error

        z = forward(flow, self.base_samples)
        error = tf.reduce_max( tf.math.abs(backward(flow, z)-self.base_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(backward(flow, z)-self.base_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for forward-reverse %s: %s (%s/%s errors)" % \
                     (msg, error, num_errors, len(self.base_samples)) )
        self.assertAlmostEqual(error, 0., precision)        
        self.assertEqual(error, error1fr)        

        z = backward(flow, self.base_samples)
        error = tf.reduce_max( abs(forward(flow, z)-self.base_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(forward(flow, z)-self.base_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for reverse-forward %s: %s (%s/%s errors)" % \
                     (msg, error, num_errors, len(self.base_samples)))   
        self.assertAlmostEqual(error, 0., precision)  
        self.assertEqual(error, error1rf)        

        z = forward(flow, self.empty_samples)
        error = tf.reduce_max( tf.math.abs(backward(flow, z)-self.empty_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(backward(flow, z)-self.empty_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for empty forward-reverse %s: %s (%s/%s errors)" % \
                (msg, error, num_errors, len(self.empty_samples)))   
        self.assertAlmostEqual(error, 0., precision)        

        z = backward(flow, self.empty_samples)
        error = tf.reduce_max( abs(forward(flow, z)-self.empty_samples) ).numpy()
        num_errors = len( np.nonzero( tf.math.abs(forward(flow, z)-self.empty_samples).numpy() > 0.0001 )[0] )
        logger.debug(" worst case error for empty reverse-forward %s: %s (%s/%s errors)" % \
                  (msg, error, num_errors, len(self.empty_samples)))   
        self.assertAlmostEqual(error, 0., precision) 


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    unittest.main()



