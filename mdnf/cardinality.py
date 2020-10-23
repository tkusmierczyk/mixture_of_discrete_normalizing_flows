# -*- coding: utf-8 -*-
""" Wrapping and unwrapping dimensions to match model cardinalities. """


import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)



def _group_category_columns(K, card):
    """ Assigns to each column 0..K-1 one category 0..card-1. """
    return np.arange(K)%card 
    # breaks backward-compatibility and does not agree with populate_categories
    #return sorted(np.arange(K)%card) 
    
    
def wrap_categories1(sample, card): 
    """ If variable has more categories than requested card wraps remaining."""
    K = sample.shape[-1]
    if K==card: 
        #logger.debug("[wrap_categories1] samples have the right shape. do nothing.")
        return sample
    sample = tf.transpose(sample) # unsorted_segment_sum acts on first dimension...
    #sample = tf.math.segment_sum(sample, group_category_columns(K, card))
    sample = tf.math.unsorted_segment_sum(sample, _group_category_columns(K, card), card)    
    sample = tf.transpose(sample)
    return sample   


def populate_categories(sample, card):
    """ Repeats the last dimension to match requested cardinality. """
    K = sample.shape[-1]
    repeat = [1]*(len(sample.shape)-1) + [int(np.ceil(card/K))]
    sample = tf.tile(sample, repeat)
    sample = sample[..., :card]    
    return sample


def pad_categories(t, K):
    """ Appends 0 in the last dimensions of tensor t until reaches K categories. """
    dims, K0 = len(t.shape), t.shape[-1]
    assert K>=K0
    if K==K0: return t
        
    padding = tf.zeros(t.shape[:-1]+[K-K0])    
    return tf.concat([t, padding], axis=-1)


def wrap_categories(sample, cardinalities, axis=-2):
    """ For each variable along axis wraps categories to match requested cardinalities.

        Remaining positions are padded with zeros.
    """
    if cardinalities is None: return sample
    N, K = sample.shape[axis], sample.shape[-1] 
   
    assert N==len(cardinalities), "number of variables (=%s) must agree with cardinalities (=%s)" % \
                                  (N, len(cardinalities))
    if (np.array(cardinalities)==K).all(): 
        logger.debug("[unwrap_categories] samples have the right shape. do nothing.")
        return sample

    s = []
    for var, card in enumerate(cardinalities):
        sample_dim = tf.gather(sample, var, axis=axis)
        sample_dim_wrapped = wrap_categories1(sample_dim, card) 
        s.append( pad_categories(sample_dim_wrapped, K) )
    s = tf.stack(s, axis=axis)
    return s


def unwrap_categories(sample, cardinalities, axis=-2):
    """ Puts 1 in each position that would wrap to the same category.
    
        For each variable along axis checks what's the target cardinality 
        and which dimensions would map down to these categories.
        For each such dimension puts 1. 
    """
    N, K = sample.shape[axis], sample.shape[-1]

    assert N==len(cardinalities), "number of variables (=%s) must agree with cardinalities (=%s)" % \
                                  (N, len(cardinalities))
    if (np.array(cardinalities)==K).all(): 
        logger.debug("[unwrap_categories] samples have the right shape. do nothing.")
        return sample

    s = []
    for var, card in enumerate(cardinalities):
        sample_dim = tf.gather(sample, var, axis=axis)
        sample_dim_wrapped = wrap_categories1(sample_dim, card) 
        sample_dim_expanded = populate_categories(sample_dim_wrapped, K) 
        s.append(sample_dim_expanded)
    s = tf.stack(s, axis=axis)    
    return s


def wrap_categories_legacy(sample, card): 
    """ If variable has more categories than requested (=card) wraps remaining.

        Legacy/reference implementation. Use wrap_categories1 instead.
    """
    if sample.shape[-1]==card: return sample
    assert sample.shape[-1]>=card 
    sample_wrapped = 0.
    for s in range(0, sample.shape[-1], card):
        e = s+card
        part = sample[..., s:e]
        padding = tf.zeros(part.shape[ :-1]+[card-part.shape[-1]], dtype=part.dtype)
        part = tf.concat([part, padding], -1) 
        sample_wrapped += part
    return sample_wrapped



