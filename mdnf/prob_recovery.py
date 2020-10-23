# -*- coding: utf-8 -*-
""" Recovering probability tables from samples or flows. """

import numpy as np
import tensorflow as tf
import itertools


def recover_prob_array(samples, vocab_sizes):
    """ Estimates probability array from samples. """
    m = np.zeros(shape=vocab_sizes)
    for sample in samples: #TODO vectorize
        m[tuple(sample)] += 1 
    m /= np.sum(m)    
    return m


def recover_prob_array_tf_one_hot(samples):
    """ Estimates probability array from one-hot encoded samples. """
    vocab_sizes = (samples.shape[-1],)*samples.shape[-2]
    samples = tf.argmax(samples, -1).numpy()
    return recover_prob_array(samples, vocab_sizes)


def generate_combinations(K, N, combinations=[[]]):
    """ Returns a list of K^N combinations 
        of length N of all possible values 0..K-1."""
    if len(combinations[0])==N: return combinations        
    new = [c+[v] for c in combinations for v in range(K)]                    
    return generate_combinations(K, N, new)


def recover_prob_array_flow(flow, base, reverse=False):
    """ Estimates probability array by pushing all the possible outputs
        backward through flow and evaluating probability from base distribution. """
    N, K = flow.N, flow.K
    run = flow.reverse if reverse==False else flow.__call__
    combinations = generate_combinations(K, N)
    sample = tf.one_hot(combinations, K)
    probs = base.prob_ext( run(sample) )

    m = np.zeros(shape=(K,)*N)
    for combination, prob in zip(combinations, probs):
        m[tuple(combination)] = prob 
    return m


def recover_prob_array_flow_samples(flow, base, 
                                    nsamples=10000, reverse=False):
    """ Estimates probability array by pushing forward 
        through flow samples from base distribution. """
    run = flow.reverse if reverse==True else flow.__call__
    base_samples = base.sample_ext(nsamples)        
    flow_output_probs = recover_prob_array_tf_one_hot( run(base_samples) )        
    return flow_output_probs


def outer_product(probs):      
    """ Returns N-dimensional joint probability array of NxK factorized categorical distributions. """      
    if len(probs.shape)==2 and probs.shape[-2]==3:
        M = np.outer(probs[0], probs[1])
        v = np.array(probs[2])
        return M[:, :, None] * v[None, None, :]
    if len(probs.shape)==2 and probs.shape[-2]==2:
        return np.outer(probs[0], probs[1])
    if len(probs.shape)==2 and probs.shape[-2]==1:
        return np.array(probs[0])
    if len(probs.shape)==1:     
        return np.array(probs)
    raise ValueError("Outer product currently implemented only for 2D and 3D arrays!")


def estimate_probabilities(samples):
    N, K = samples.shape[-2:]
    combinations = generate_combinations(K, N)
    probs = []
    for v in combinations:
        mask = tf.one_hot(v, K, dtype=samples.dtype)
        p1 = tf.reduce_sum( tf.reduce_prod( tf.reduce_sum(samples*mask, -1), -1) ) / samples.shape[0]
        probs.append(p1)
    return tf.stack(probs)


def kl_divergences(base, flow, target=None, EPS=1e-31):            
    """ Returns a triple (KL(q|p), KL(p|q), probability table of q). """
    if target is None:
        flow_output_probs = None
        kl, kl2 = float('nan'), float('nan')
    else:
        flow_output_probs = recover_prob_array_flow(flow, base)
        kl =  np.sum(flow_output_probs * (np.log(flow_output_probs+EPS)-np.log(target+EPS)))
        kl2 = np.sum(target * (-np.log(flow_output_probs+EPS)+np.log(target+EPS)))        
    return kl, kl2, flow_output_probs  



