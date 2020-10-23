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
""" Operations on one-hot encoded vectors.

 Based on Edward2 code by Dustin Tran from 
 https://github.com/google/edward2/
 Fixes numerical stability of one_hot_add operation.
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np


def one_hot_argmax(inputs, temperature, axis=-1):
  """Returns one-hot of argmax with backward pass set to softmax-temperature."""
  vocab_size = inputs.shape[-1]
  hard = tf.one_hot(tf.argmax(inputs, axis=axis),
                    depth=vocab_size,
                    axis=axis,
                    dtype=inputs.dtype)
  soft = tf.nn.softmax(inputs / temperature, axis=axis)
  outputs = soft + tf.stop_gradient(hard - soft)
  return outputs


def one_hot_minus(inputs, shift): 
  """Performs (inputs - shift) % vocab_size in the one-hot space."""
  inputs = tf.convert_to_tensor(inputs)
  shift = tf.cast(shift, inputs.dtype)
  vocab_size = inputs.shape[-1]
  # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
  # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
  # "shifts" the inputs batch element by the corresponding shift batch element.
  shift_matrix = tf.stack([tf.roll(shift, i, axis=-1)
                           for i in range(vocab_size)], axis=-2)
  outputs = tf.einsum('...v,...uv->...u', inputs, shift_matrix)
  return outputs


def one_hot_add(inputs, shift):
    """ Fixed version using one_hot_minus. """
    inv_shift = tf.reverse(shift[..., 1:], [-1])
    shift = tf.concat([shift[..., :1], inv_shift], -1)
    return one_hot_minus(inputs, shift)



def one_hot_multiply(inputs, scale):
  """Performs (inputs * scale) % vocab_size in the one-hot space.
  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to scale the corresponding one-hot vector in
      inputs. Soft values perform a "weighted scale": for example,
      scale=[0.2, 0.3, 0.5] performs a linear combination of
      0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.
  Returns:
    Tensor of same shape and dtype as inputs.
  """
  inputs = tf.convert_to_tensor(inputs)
  scale = tf.cast(scale, inputs.dtype)
  batch_shape = inputs.shape[:-1].as_list()
  vocab_size = inputs.shape[-1]
  if isinstance(vocab_size, tf1.Dimension):
    vocab_size = vocab_size.value
  # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
  # batched vocab_size x vocab_size matrix represents scaling inputs by i.
  permutation_matrix = tf.math.floormod(
      tf.tile(tf.range(vocab_size)[:, tf.newaxis], [1, vocab_size]) *
      tf.range(vocab_size)[tf.newaxis], vocab_size)
  permutation_matrix = tf.one_hot(permutation_matrix, depth=vocab_size, axis=-1)
  # Scale the inputs according to the permutation matrix of all possible scales.
  scaled_inputs = tf.einsum('...v,avu->...au', inputs, permutation_matrix)
  scaled_inputs = tf.concat([tf.zeros(batch_shape + [1, vocab_size]),
                             scaled_inputs[..., 1:, :]], axis=-2)
  # Reduce rows of the scaled inputs by the scale values. This forms a
  # weighted linear combination of scaling by zero, scaling by one, and so on.
  outputs = tf.einsum('...v,...vu->...u', scale, scaled_inputs)
  return outputs



def py_multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n (in Python).
  Implements extended Euclidean algorithm.
  Args:
    a: int-like np.ndarray.
    n: int.
  Returns:
    Multiplicative inverse as an int32 np.ndarray with same shape as a.
  """
  batched_a = np.asarray(a, dtype=np.int32)
  batched_inverse = []
  for a in np.nditer(batched_a):
    inverse = 0
    new_inverse = 1
    remainder = n
    new_remainder = a
    while new_remainder != 0:
      quotient = remainder // new_remainder
      (inverse, new_inverse) = (new_inverse, inverse - quotient * new_inverse)
      (remainder, new_remainder) = (new_remainder,
                                    remainder - quotient * new_remainder)
    if remainder > 1:
      return ValueError(
          'Inverse for {} modulo {} does not exist.'.format(a, n))
    if inverse < 0:
      inverse += n
    batched_inverse.append(inverse)
  return np.asarray(batched_inverse, dtype=np.int32).reshape(batched_a.shape)


def multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n.
  Args:
    a: Tensor of shape [..., vocab_size]. It denotes an integer in the one-hot
      space.
    n: int Tensor of shape [...].
  Returns:
    Tensor of same shape and dtype as a.
  """
  a = tf.convert_to_tensor(a)
  n = tf.convert_to_tensor(n)
  vocab_size = a.shape[-1]
  if isinstance(vocab_size, tf1.Dimension):
    vocab_size = vocab_size.value
  a_dtype = a.dtype
  sparse_a = tf.argmax(a, axis=-1)
  # TODO(trandustin): Switch to tf.function.
  sparse_outputs = tf1.py_func(
      py_multiplicative_inverse, [sparse_a, n], tf.int32)
  sparse_outputs.set_shape(sparse_a.shape)
  outputs = tf.one_hot(sparse_outputs, depth=vocab_size, dtype=a_dtype)
  return outputs



def one_hot_multiply(inputs, scale):
  """Performs (inputs * scale) % vocab_size in the one-hot space.
  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to scale the corresponding one-hot vector in
      inputs. Soft values perform a "weighted scale": for example,
      scale=[0.2, 0.3, 0.5] performs a linear combination of
      0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.
  Returns:
    Tensor of same shape and dtype as inputs.
  """
  inputs = tf.convert_to_tensor(inputs)
  scale = tf.cast(scale, inputs.dtype)
  batch_shape = inputs.shape[:-1].as_list()
  vocab_size = inputs.shape[-1]
  if isinstance(vocab_size, tf1.Dimension):
    vocab_size = vocab_size.value
  # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
  # batched vocab_size x vocab_size matrix represents scaling inputs by i.
  permutation_matrix = tf.math.floormod(
      tf.tile(tf.range(vocab_size)[:, tf.newaxis], [1, vocab_size]) *
      tf.range(vocab_size)[tf.newaxis], vocab_size)
  permutation_matrix = tf.one_hot(permutation_matrix, depth=vocab_size, axis=-1)
  # Scale the inputs according to the permutation matrix of all possible scales.
  scaled_inputs = tf.einsum('...v,avu->...au', inputs, permutation_matrix)
  scaled_inputs = tf.concat([tf.zeros(batch_shape + [1, vocab_size]),
                             scaled_inputs[..., 1:, :]], axis=-2)
  # Reduce rows of the scaled inputs by the scale values. This forms a
  # weighted linear combination of scaling by zero, scaling by one, and so on.
  outputs = tf.einsum('...v,...vu->...u', scale, scaled_inputs)
  return outputs



