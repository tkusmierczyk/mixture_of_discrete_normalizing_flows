# -*- coding: utf-8 -*-

import unittest

import flows
import flows_mixture

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
        K = 23 
        B = 3
        self.N = N
        self.K = K
        self.B = B

        base = tfp.distributions.OneHotCategorical(logits=np.random.randn(N,B,K))
        logger.debug("[init] base=%s" % (base,) )
        self.base_samples = tf.cast(base.sample(256), dtype='float32')
        logger.debug("[init] samples=%s (%s)" % (self.base_samples.shape, self.base_samples.dtype))

        self.mixture_specs = [ #single-layer mixtures to be tested
                    [("M",[12, 16]), ("M",[5]), ("M",[3])],
                    [("F", None), ("F", None), ("F", None)], 
                    [("M",[12, 16]), ("F", None), ("M",[5])],
                    [("M",[12, 16]), ("F", None), ("F",None)],
                    [("M",[12, 16]), ("F", None), ("M",[11,13])],
                    ]

    def setUp(self):
        np.random.seed(13)
        tf.random.set_seed(13)
   
    def test_discrete_mixture_reverse(self):
        """Tests if samples passed in the reverse direction consist of reversed samples from individual flows."""
        N, K, B = self.N, self.K, self.B
        base_samples = self.base_samples

        for i, specs in enumerate(self.mixture_specs):
            logger.debug("#"*30+" [%i/%i] testing reverse of mixture of %s" % \
                         (i+1, len(self.mixture_specs), specs))
            flows1 = flows.parse_layers_specification(specs, N, K, temperature=5.0)
            mixture = flows_mixture.DiscreteFlowsMixture(N,K,B,flows=flows1, temperature=5.0) #mixture of the flows
            
            # test if reverse is composed out of reversed of individual 
            out_samples = base_samples[...,0,:] # just any 'output' x
            inv_samples = mixture.reverse(out_samples) # reverse transformation to u
            for b, f in enumerate(flows1): 
                self.assertTrue( bool(tf.reduce_all(f.reverse(out_samples)==inv_samples[...,b,:])), \
                                 "reverse of %i-th flow" % b)

    def test_discrete_mixture_forward_nonmasked(self):
        """Tests if forward pass without masking results in invalid samples having exactly B ones per dimension."""
        N, K, B = self.N, self.K, self.B
        base_samples = self.base_samples

        for i, specs in enumerate(self.mixture_specs):
            logger.debug("#"*30+" [%i/%i] testing forward (no mask) of mixture of %s" % \
                         (i+1, len(self.mixture_specs), specs))
            flows1 = flows.parse_layers_specification(specs, N, K, temperature=5.0)
            mixture = flows_mixture.DiscreteFlowsMixture(N, K, B, flows=flows1, temperature=5.0) #mixture of the flows
            
            out_samples = mixture(base_samples)
            self.assertTrue( bool(tf.reduce_all(tf.reduce_sum(out_samples, -1)==B)), 
                             "non-masked transformation through mixture=%s has %i ones per dim." % (specs, B))
            
    def test_discrete_mixture_forward(self):
        """Tests if forward pass with a mask matches outputs from the mask-selected flows."""
        N, K, B = self.N, self.K, self.B
        base_samples = self.base_samples

        for i, specs in enumerate(self.mixture_specs):
            logger.debug("#"*30+" [%i/%i] testing forward of mixture of %s" % \
                         (i+1, len(self.mixture_specs), specs))

            flows1 = flows.parse_layers_specification(specs, N, K, temperature=5.0)
            mixture = flows_mixture.DiscreteFlowsMixture(N, K, B, flows=flows1, temperature=5.0) #mixture of the flows

            for b in range(B):
                flow1_out_samples = flows1[b](base_samples[...,b,:])
                mask = tf.one_hot([b], B)[None,:][...,None]
                out_samples = mixture(base_samples, mask=mask)

                self.assertTrue( bool(tf.reduce_all(tf.reduce_sum(out_samples, -1)==1)), 
                                 "masked samples passed through mixture=%s have exactly one 1 per dim." % (specs))

                self.assertTrue( tf.reduce_all(flow1_out_samples == out_samples), 
                                 "choosing outputs from %i-th flow" % b)

            
    def test_discrete_mixture_forward_random(self):
        """Tests if forward pass with a randomly generated mask matches outputs from the selected flows."""
        N, K, B = self.N, self.K, self.B
        base_samples = self.base_samples

        for i, specs in enumerate(self.mixture_specs):
            logger.debug("#"*30+" [%i/%i] testing forward of mixture of %s" % \
                         (i+1, len(self.mixture_specs), specs))

            flows1 = flows.parse_layers_specification(specs, N, K, temperature=5.0)
            mixture = flows_mixture.DiscreteFlowsMixture(N, K, B, flows=flows1, temperature=5.0) #mixture of the flows

            bs = np.random.choice(range(B), base_samples.shape[0]) # assign each sample to a flow from mixture
            mask = tf.one_hot(bs, B)[...,None,:,None] # create mask
            out_samples = mixture(base_samples, mask=mask)

            self.assertTrue( bool(tf.reduce_all(tf.reduce_sum(out_samples, -1)==1)), 
                                 "masked samples passed through mixture=%s have exactly one 1 per dim." % (specs))
            
            for b in range(B): 
                bth_flow_sample_indices = np.nonzero(bs==b)[0]
                bth_flow_out_samples = tf.gather(out_samples, bth_flow_sample_indices, -2)

                # pass assigned samples through individual flow
                flow1_out_samples = flows1[b](base_samples[...,b,:])                 
                bth_flow1_out_samples = tf.gather(flow1_out_samples, bth_flow_sample_indices, -2)
                
                self.assertTrue(bth_flow_out_samples.shape==bth_flow1_out_samples.shape, 
                 "mask selects correct number of samples assigned to %d-th flow" % b)
                self.assertTrue(bool(tf.reduce_all(bth_flow_out_samples==bth_flow1_out_samples)),
                 "check if results from a mixture match results from individual %d-th flow" % b)

    def test_temperature_update(self):
        """Tests if temperature updates affect gradients."""
        flow = flows_mixture.DiscreteFlowsMixture(self.N, self.K, self.B,
                temperature=10., single_flow_layers=[("M",[16, 16, 8]), ("F", None), ("M",[])]) 

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



if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    unittest.main()



