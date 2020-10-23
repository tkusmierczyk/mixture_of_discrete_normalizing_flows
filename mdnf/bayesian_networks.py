# -*- coding: utf-8 -*-
""" Joint log-probability evaluation for arbitrary bayesian networks. """


import numpy as np
import tensorflow as tf
from itertools import product

import tensorflow_probability as tfp
tfd = tfp.distributions

import logging
logger = logging.getLogger(__name__)

from pgmpy.readwrite import BIFReader

from cardinality import wrap_categories1


def cpds_to_variables(cpds):
    """ Returns a dictionary {variable: list-of-possible-values} 
        extracted from a list of pgmpy CPDs."""
    variable2values = {}
    for cpd in cpds:
        var2val = dict((name, ["val=%i"%c for c in range(card)]) 
                   for name, card in zip(cpd.variables, cpd.cardinality))
        if cpd.state_names:
            var2val = cpd.state_names 
        variable2values.update(var2val)
    return variable2values


def cpd_to_prob_array(cpd, verbose=True):
    """ Converts a pgmpy CPD into a probability multidimensional np array. """
    if verbose:
        logger.info(cpd)
        variables = cpd.variables
        var2val = cpds_to_variables([cpd])
        logger.debug("variable2position=%s" % var2val)
    
    indexes = np.array(list(product(*[range(i) for i in cpd.cardinality])))
    values = cpd.get_values().reshape(-1)
    probs = np.zeros(cpd.cardinality)
    for pos, v in zip(indexes, values):
        if verbose:
            header = ", ".join( "%s=%s(=%i)" % (n, var2val[n][p], p) 
                                for n, p in zip(variables, pos) )
            logger.debug("%s => p=%s" % (header, v))
        probs[tuple(pos)] = v    
    return probs



def calc_sample_prob(sample, probs):
    """ Returns probabilities of one-hot encoded samples. 
        
        Args:
            sample  One-hot encoded sample.
                    Shape = batch x nvariables x max-cardinality.
            probs  Conditional probability tensor.
                   First dimension (==0) chooses conditioning variable.
            
    """
    cardinalities = probs.shape
    N = len(cardinalities)
    assert len(sample.shape)==3, "sample must have shape = batch x nvariables x max-cardinality"
    assert sample.shape[-2]==N, "sample must have nvariables=%s but has shape=%s" % (N, sample.shape)
    assert sample.shape[-1]>=max(cardinalities), "samples must have >= max(cardinalities==probs.shape)"
    
    p = probs[None, ...] # expand by batch size on dim 0
    for var, card in enumerate(cardinalities):

        shp = np.ones( (N,), dtype=int) # ones in all dimensions 
        shp[var] = card # apart from the current one
        shp = [-1]+list(shp) # with batch size on dim 0
        
        sample_var = wrap_categories1(sample[..., var, :], card)
        p = p * tf.reshape(sample_var, shp) 
        
    p = tf.reduce_sum(p, range(1, N+1)) 
    return p


def calc_sample_log_relaxed_prob(sample, probs, temperature, eps=1e-31):
    """ Returns log-probabilities of one-hot encoded samples. 
        First, conditioned variables are selected from probs, 
        then log-probability of the conditioning variable (==0) is obtained with RelaxedOneHotCategorical.        
    """
    cardinalities = probs.shape
    N = len(cardinalities)
    assert len(sample.shape)==3, "sample must have shape = batch x nvariables x max-cardinality"
    assert sample.shape[-2]==N, "sample must have nvariables=%s but has shape=%s" % (N, sample.shape)
    assert sample.shape[-1]>=max(cardinalities), "samples must have >= max(cardinalities==probs.shape)"
    
    # select dimensions according to all variables apart from the conditioning one (=no 0)
    p = tf.constant(probs[None, ...], dtype=sample.dtype) # expand by batch size on dim 0
    for var in range(1, N): 
        card = cardinalities[var]
        
        shp = np.ones( (N,), dtype=int) # ones in all dimensions 
        shp[var] = card # apart from the current one
        shp = [-1]+list(shp) # with batch size on dim 0
        
        sample_var = wrap_categories1(sample[..., var, :], card)
        p = p * tf.reshape(sample_var, shp)
    p = tf.reduce_sum(p, range(1+1, N+1))    
    
    # relaxed log_prob density 
    d = tfd.RelaxedOneHotCategorical(temperature=temperature, probs=p)
    return d.log_prob(sample[..., 0, :]+eps)  


def marginalize(prob_array, keep_dim_no):
    axes = set(range(len(prob_array.shape)))
    axes.remove(keep_dim_no)
    return tf.reduce_sum(prob_array, list(axes))


def as_tensor(positions, probabilities, drop1=True):
    """ Returns an array with probabilities stored in selected positions.

        Args:
            positions  An array having in each row a vector [dim1, ..., dimN] 
                       telling where to store a respective probability.
            probabilities  A vector with probabilities.
            drop1  If True, position dimensions consisting of only one value will be dropped
                   (if cardinality(dim_i)==1 then dim_i will flattened).
    """
    if drop1:
        active_cols = np.max(positions, 0)-np.min(positions, 0) > 0
        logger.debug("keeping %i out of %i cols" % (np.sum(active_cols), len(active_cols)))
        positions = positions[:, active_cols]

    probs_tensor = np.zeros( tuple(np.max(positions, 0)+1) )
    for pos, prob in zip(positions, probabilities):
        probs_tensor[tuple(pos)] = prob
    return probs_tensor


# Used to remove inf-s due to zero probabilities
EPS = 10e-16


class BayesianNetworkVI:
    """ Joint log-probability evaluation for arbitrary bayesian networks. """
    
    def __init__(self, evidence, bif_path=None, model=None):
        """
            Args:
                evidence  A dictionary with observations, e.g., {node-name: node-value}. 
                model / bif_path  PGMPY model specification or .bif file path.
        """

        assert bif_path is not None or model is not None, \
                "You must set either pgmpy model or bif_path!"
        if model is None:
            logger.debug("[BayesianNetworkVI] reading from %s" % bif_path)
            reader = BIFReader(bif_path)
            model = reader.get_model()
        self._model = model
        self._evidence = evidence
        
        latent_variables = sorted( set(v for cpd in self._model.get_cpds() 
                                         for v in cpd.variables).difference(evidence) )
        observed_variables = sorted(evidence)
        self._no2var = dict(enumerate(latent_variables+observed_variables))
        self._var2no = dict((var, no) for no, var in self._no2var.items())  
        logger.debug("[BayesianNetworkVI] mapping(id->variable)=%s" % self._no2var)      

        variable2values = cpds_to_variables(self._model.get_cpds())
        self._variable2values = dict( (var, dict( (v, i) for i, v in enumerate(vals) )) 
                                             for var, vals in variable2values.items()) 
        
        self._prob_arrays = [cpd_to_prob_array(cpd)+EPS for cpd in self._model.get_cpds()] #!TODO renormalization

        logger.info("[BayesianNetworkVI] %i vars with max cardinality=%i => enumeration size=%i" % 
                    (len(variable2values), self.cardinality, self.enumeration_size))       

    @property
    def variables(self):
        """ Information about ordering of variables and variable values. """
        return dict( ((n, v), self._variable2values.get(v,"ERROR")) for v, n in self._var2no.items())

    @property
    def enumeration_size(self):
        #return np.prod(list(self.var2cardinality.values()))
        prev = 1.
        for v in self.var2cardinality.values():
            if v*prev < prev:
                logger.warning("WARNING: overflow when calculating enumeration size!")
                return float("inf")
            prev = v*prev
        return prev

    @property
    def cardinality(self):
        """ Max cardinality of all variables. """
        return max(self.var2cardinality.values())

    @property
    def cardinalities(self):
        return [len(v) for k, v in sorted(self.variables.items())]

    @property
    def var2cardinality(self):
        """ A dictionary {variable_name: variable_cardinality}. """
        return dict( (var, len(val)) for var, val in self._variable2values.items() )

    @property
    def N(self):
        """ Number of variables. """
        return len(self._no2var)

    def posteriors_via_enumeration(self, evidence=None):
        """ Returns all possible values of the variables 
            (that match evidence) and respective probabilities.

            Ordering aggrees with the one in self.variables.
        """
        if evidence is None: evidence = self._evidence

        enumeration = self.enumerate_variables(evidence)
        samples = tf.one_hot(enumeration, self.cardinality)
        probs = tf.math.exp(self.log_prob(samples))
        probs /= tf.reduce_sum(probs)

        return enumeration, probs

    def enumerate_variables(self, evidence={}):
        """ Returns all possible sets of values that match evidence. """
        if self.enumeration_size > 1000000:
            logger.warning("Enumeration size is %s. May be infeasible!" % self.enumeration_size)

        variables = list(self._no2var.values()) # ordered variables
        cards = [self.var2cardinality[var] for var in variables] 
        enumeration = np.array(list(product(*[range(i) for i in cards])))
        logger.debug("[enumerate_variables] all possible configs = %s" % len(enumeration))
        
        if evidence is None: evidence = self._evidence
        if len(evidence)<=0: return enumeration

        evidence = dict((self._var2no[var], self._variable2values[var].get(val, val)) 
                                                    for var, val in evidence.items())
        logger.debug("[enumerate_variables] fixed evidence = %s" % evidence)
        for col, val in evidence.items():
            enumeration = enumeration[enumeration[:, col]==val, :]
        logger.debug("[enumerate_variables] configs matching evidence = %s" % len(enumeration))
        return enumeration

    def set_evidence(self, sample=None, evidence=None):
        """ Replaces variables in sample (one-hot encoded tf tensor)
            according to evidence dictionary {variable_name: value_name/value_int}. 

            Args:
                sample  A tf tensor of size batch x (#variables or #unobserved variables) x max-cardinality 
                        If sample is None all values will be set according to evidence.
        """
        if sample is None:            
            sample = tf.zeros( (1, self.N, self.cardinality) )
            assert len(evidence)==self.N, "If sample is not set, evidence must cover all %i variables." % self.N
        assert sample.shape[-2]==len(self._no2var)-len(self._evidence) or sample.shape[-2]==len(self._no2var), \
                "sample should have size =#unobserved_variables (=not in evidence) or =#variables"
        if evidence is None:        
            evidence = self._evidence

        #order evidence according to dimensions order
        evidence = sorted(evidence.items(), key=lambda k2v: self._var2no[k2v[0]]) 

        for variable, value in evidence:
            var_no = self._var2no[variable]
            value_no = self._variable2values[variable].get(value, value)

            var_shape = sample.shape[:-2] + [1] + sample.shape[-1:]
            var_tf = tf.broadcast_to(tf.one_hot(value_no, depth=sample.shape[-1]), var_shape)
            sample = tf.concat([ sample[...,:var_no,:], var_tf, sample[...,(var_no+1):,:] ], axis=-2)    

        return sample      

    def log_prob(self, sample):
        """ Returns joint log probability for (one-hot encoded) variable values.

            Args:
                sample  A tf tensor of size batch x nvariables x max-cardinality 
                        of one-hot encoded values for variables ordered according to self.variables.
        """
        assert len(sample.shape)==3, \
                "sample must have shape = batch x nvariables x max-cardinality"
        assert sample.shape[-1]>=self.cardinality, \
                "sample number of categories must >= cardinality=%i" % self.cardinality

        logprobs = 0.
        for probs, cpd in zip(self._prob_arrays, self._model.get_cpds()): #sum over blocks
            var_nos = [self._var2no[name] for name in cpd.variables]
            var_sample = tf.gather(sample, var_nos, axis=-2)
            var_probs = calc_sample_prob(var_sample, probs)
            logprobs += tf.math.log(var_probs)
        return logprobs

    def log_prob_evidence(self, sample, evidence=None):
        """ Returns log probability for observed values."""
        if evidence is None:        
            evidence = self._evidence

        logprobs = 0.
        for probs, cpd in zip(self._prob_arrays, self._model.get_cpds()): #sum over blocks
            if cpd.variables[0] not in evidence: continue # include only probs for observed data 
            var_nos = [self._var2no[name] for name in cpd.variables]
            var_sample = tf.gather(sample, var_nos, axis=-2)            
            logprobs += tf.math.log( calc_sample_prob(var_sample, probs) )
        return logprobs

    def log_relaxed_prob_priors(self, sample, temperature, evidence=None, eps=1e-31):
        """ Returns log probability for unobserved values using RelaxedOneHotCategorical distribution."""
        if evidence is None:        
            evidence = self._evidence

        logprobs = 0.
        for probs, cpd in zip(self._prob_arrays, self._model.get_cpds()): #sum over blocks
            if cpd.variables[0] in evidence: continue # include only probs for unobserved data 
            var_nos = [self._var2no[name] for name in cpd.variables]
            var_sample = tf.gather(sample, var_nos, axis=-2)            
            logprobs += calc_sample_log_relaxed_prob(var_sample, probs, temperature, eps=eps)
        return logprobs
    

