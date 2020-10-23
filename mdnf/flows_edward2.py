# -*- coding: utf-8 -*-
# Copyright 2019 The Edward2 Authors.
# Modifications copyright (C) 2020 Tomasz Kusmierczyk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
 Discrete autoregressive and bipartite flows.
 Based on Edward2 code by Dustin Tran from 
 https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py

 1) Uses one_hot operations instead of the original ones.
 2) Simplifies DiscreteAutoregressiveFlow by dropping sigma transformation.
 3) Implements DiscreteAutoregressivePartialFlow.
 4) Breaks the dependency on Edward2.
"""


import tensorflow as tf
import one_hot as utils
import numpy as np


class DiscreteAutoregressiveFlow(tf.keras.layers.Layer):
  """ A discrete reversible layer (with only shift operation supported). """

  def __init__(self, layer, temperature, **kwargs):
    """Constructs flow.
    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    return super(DiscreteAutoregressiveFlow, self).__call__(inputs, *args, **kwargs)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2]
    if length is None:
      raise NotImplementedError('length dimension must be known.')
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].
    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs[..., tf.newaxis, :]
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(inputs, paddings=[[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., 0:1, :]
      loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      outputs =  utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].
    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs,
        paddings=[[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      new_outputs =  utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    net = self.layer(inputs, **kwargs)
    if net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    outputs =  utils.one_hot_add(loc, scaled_inputs)
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


class DiscreteAutoregressivePartialFlow(DiscreteAutoregressiveFlow):
  """ A discrete reversible layer with shift transformation 
      acting on subset of positions.
  """

  def __init__(self, K, categories, layer, temperature, **kwargs):
    super().__init__(layer, temperature, **kwargs)

    subset_K = len(categories)
    assert subset_K <= K
    
    shuffling = np.arange(K)
    np.random.shuffle(shuffling)
    shuffling = np.array(list(categories)+[v for v in shuffling if v not in categories])

    inverted_shuffling = np.arange(K)
    inverted_shuffling[shuffling] = np.arange(K)

    self.shuffling = shuffling
    self.inverted_shuffling = inverted_shuffling
    self.subset_K = subset_K

  def __call__(self, inputs, *args, **kwargs):
    return super(DiscreteAutoregressivePartialFlow, self).__call__(inputs, *args, **kwargs)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2]
    if length is None:
      raise NotImplementedError('length dimension must be known.')

    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    inputs = new_inputs[..., tf.newaxis, :]
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(inputs, paddings=[[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == self.subset_K: #!
      loc = net
      loc = loc[..., 0:1, :]
      loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)

      # operate on subset:
      x = inputs 
      x = tf.gather(x, self.shuffling, axis=-1)
      x1 = x[..., : self.subset_K]
      #x1 = super().call(x1)
      x1 = utils.one_hot_minus(x1, loc) #outputs =  utils.one_hot_minus(inputs, loc)
      x2 = x[...,self.subset_K: ]
      x = tf.concat([x1,x2], axis=-1)
      x = tf.gather(x, self.inverted_shuffling, axis=-1)
      outputs =  x

    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs,
        paddings=[[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == self.subset_K: #!
      loc = net
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)

      # operate on subset:
      x = inputs
      x = tf.gather(x, self.shuffling, axis=-1)
      x1 = x[..., : self.subset_K]
      #x1 = super().call(x1)
      x1 = utils.one_hot_minus(x1, loc) #new_outputs =  utils.one_hot_minus(inputs, loc)
      x2 = x[...,self.subset_K: ]
      x = tf.concat([x1,x2], axis=-1)
      x = tf.gather(x, self.inverted_shuffling, axis=-1)
      new_outputs =  x

    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    net = self.layer(inputs, **kwargs)
    if net.shape[-1] == self.subset_K: #!
      loc = net
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast( utils.one_hot_argmax(loc, self.temperature), inputs.dtype)

    # operate on subset:
    x = inputs
    x = tf.gather(x, self.shuffling, axis=-1)
    x1 = x[..., : self.subset_K]
    #x1 = super().call(x1)
    x1 = utils.one_hot_add(loc, x1)
    x2 = x[...,self.subset_K: ]
    x = tf.concat([x1,x2], axis=-1)
    x = tf.gather(x, self.inverted_shuffling, axis=-1)
    outputs =  x

    return outputs


class DiscreteBipartiteFlow(tf.keras.layers.Layer):


  def __init__(self, layer, mask, temperature, **kwargs):
    """Constructs flow.
    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      mask: binary Tensor of shape `[length]` forming the bipartite assignment.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteBipartiteFlow, self).__init__(**kwargs)
    self.layer = layer
    self.mask = mask
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf1.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteBipartiteFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(DiscreteBipartiteFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for bipartite generation."""
    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs    
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == self.vocab_size:
      loc = net
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      masked_outputs = (1. - mask) * utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = masked_inputs + masked_outputs
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass for the inverse bipartite transformation."""
    if not self.built:
      self._maybe_build(inputs)

    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    masked_outputs = (1. - mask) * utils.one_hot_add(loc, scaled_inputs)
    outputs = masked_inputs + masked_outputs
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


