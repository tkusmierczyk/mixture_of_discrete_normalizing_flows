# Reliable Categorical Variational Inference with Mixture of Discrete Normalizing Flows 


## Quick start

The most basic but fully-functional implementation of mixture of discrete flows (assuming factorized posterior) can be previewed in [flows_factorized_mixture.py](mdnf/flows_factorized_mixture.py). The implementation is used by [VAEFlowsBasic.ipynb](notebooks/VAEFlowsBasic.ipynb) - a demonstration of MDNF for VAE (amortized inference example).


## Publication

The code was used and is necessary to reproduce results from the paper:

T. Kuśmierczyk, A. Klami: **Reliable Categorical Variational Inference with Mixture of Discrete Normalizing Flows** [(arXiv preprint)](https://arxiv.org/pdf/2006.15568.pdf)


## Code organization

The code was implemented to support maximum flexibility via modularization, which makes various experiments possible, but at the cost of sacrificing efficiency. It separates the construction of individual flows from the construction of base distributions, from the creation of mixtures and inference algorithms. An exception is the [flows_factorized_mixture.py](mdnf/flows_factorized_mixture.py) that includes sampling from the base, location-shift transformation, and probability evaluation altogether in a single file. 


## Description of files

1. [notebooks](notebooks) - Jupyter notebooks illustrating use of MDNF with various models (the same code used later for experiments):
 * [BayesianNetworkFlows.ipynb](notebooks/BayesianNetworkFlows.ipynb) - BNs with MDNF
 * [BayesianNetworkFlowsFactorized.ipynb](notebooks/BayesianNetworkFlowsFactorized.ipynb) - BNs with MDNF (configuration assuming factorized posterior)
 * [BayesianNetworkConcrete.ipynb](notebooks/BayesianNetworkConcrete.ipynb) - BNs with Gumbel-Softmax (relaxed priors)
 
 * [VAEFlows.ipynb](notebooks/VAEFlows.ipynb) - VAEs with MDNF
 * [VAEFlowsBasic.ipynb](notebooks/VAEFlowsBasic.ipynb) - VAEs with MDNF (an implementation using the simplified implementation of factorized flows)
 * [VAEConcrete.ipynb](notebooks/VAEConcrete.ipynb) - VAEs with Gumbel-Softmax relaxations (both with and without ST; also for improper objectives apart from the Jang's loss)
 * [VAEConcreteJang.ipynb](notebooks/VAEConcreteJang.ipynb) - VAEs with Gumbel-Softmax relaxation using Jang's approximate objective  

 * [GaussianMixture.ipynb](notebooks/GaussianMixture.ipynb) - GMMs with MDNF
 
 * [PartialFlows.ipynb](notebooks/PartialFlows.ipynb) - Comparison of partial vs. location-scale flows
 * [VAEFlowsBasic_Variance.ipynb](notebooks/VAEFlowsBasic_Variance.ipynb) - Estimation of variance of MDNF objective for a VAE

2. [mdnf](mdnf) - main files implementing flows, mixtures, base distributions, inference etc.:
 * [flows_factorized_mixture.py](mdnf/flows_factorized_mixture.py) - The most basic (single file) implementation of mixture of discrete flows assuming factorized posterior. It implements sampling from delta base distribution, shift-only transformation, mixing and probability evaluation with MDNF.
 * [one_hot.py](mdnf/one_hot.py) - Operations on one-hot encoded vectors. 
 
 * [flows_mixture.py](mdnf/flows_mixture.py) - Mixture of discrete normalizing flows. 
 * [inference.py](mdnf/inference.py) - Variational inference algorithms for discrete normalizing flows. 
  
 * Base distributions:
   * [base_mixtures.py](mdnf/base_mixtures.py) - Base for mixture of categorical distributions. 
   * [base_categorical.py](mdnf/base_categorical.py) - Factorized categorical distribution. 
   * [base_constructors.py](mdnf/base_constructors.py) - Creating base mixtures of categorical distributions. 
   
 * Individual discrete flows:
   * [flows_transformations.py](mdnf/flows_transformations.py) - Networks calculating transformations for discrete flows. 
   * [flows.py](mdnf/flows.py) - Basic flows.
   * [flows_factorized.py](mdnf/flows_factorized.py) - Discrete flows for factorized distributions.  
   * [flows_edward2_made.py](mdnf/flows_edward2_made.py) - Masked autoencoders.  
   * [flows_edward2.py](mdnf/flows_edward2.py) - Discrete autoregressive flows.

 * Models:
   * [bayesian_networks.py](mdnf/bayesian_networks.py) - Evaluation of joint probability of observations and latent variables for arbitrary Bayesian networks. 
   * [gmvi.py](mdnf/gmvi.py) - Variational Gaussian Mixture using discrete normalizing flows. 
   * [cardinality.py](mdnf/cardinality.py) - Wrapping and unwrapping dimensions to match model cardinalities.

 * Auxiliary:
   * [prob_recovery.py](mdnf/prob_recovery.py) - Recovering probability tables from samples or flows. 
   * [aux.py](mdnf/aux.py) - General auxiliary functions. 
   * [time_profiling.py](mdnf/time_profiling.py) - Auxiliary functions for measuring time. 

 * Unit tests:
   * [one_hot_test.py](mdnf/one_hot_test.py) 
   * [flows_test.py](mdnf/flows_test.py) 
   * [flows_mixture_test.py](mdnf/flows_mixture_test.py) 
   

## Specification of dependencies

Please start by installing *requirements.txt*. 
In case of problems consult the following:

The code was tested with Python 3.7.4 (on a Linux platform),
using *tensorflow 2.2.0* and *tensorflow_probability 0.9.0*
(can be installed with `pip install tensorflow==2.2.0 tensorflow_probability=0.9.0`).
It also requires *numpy*, *pandas*, *sklearn* and *scipy*,
that can be installed with `pip install numpy pandas sklearn scipy`,
 but are also available by default in for example,
 [python Anaconda distributions](https://www.anaconda.com/products/individual).
Potential problems with *scipy 1.4.1* can be solved by downgrading it to version 1.2.1 with 
`pip install scipy==1.2.1`.

Notebooks *.ipynb* can be previewed using *Jupyter Notebook* and run from a command line with *runipy*. 
Visualizing results requires *matplotlib* and *seaborn* to be available (`pip install matplotlib seaborn`).

Parts of the code for Bayesian networks require PGMPY
(`pip install pgmpy==0.1.10`) and 
code for Gaussian mixture models builds on 
[Python codes implementing algorithms described in Bishop's book](https://github.com/ctgk/PRML) 
(can by installed with `git clone https://github.com/ctgk/PRML; cd PRML; python setup.py install`). 

Finally, the code comparing partial and location-scale flows uses 
[Edward2](https://github.com/google/edward2) that can be installed with 
`pip install "git+https://github.com/google/edward2.git#egg=edward2"`

