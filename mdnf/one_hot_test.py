# -*- coding: utf-8 -*-

import unittest

import numpy as np
import tensorflow as tf

import one_hot


import logging
logger = logging.getLogger(__name__)



class TestOneHot(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)
        #tf.keras.backend.set_floatx('float64')

    def setUp(self):
        np.random.seed(13)
        tf.random.set_seed(13)

    def assertAllEqual(self, a, b, msg=""):
        self.assertTrue(tf.reduce_all(a==b), msg)

    def test_one_hot_add(self):
        """Test one_hot_add (if max value position moves by shift)."""
        K = 8
        vals = np.array([1.,3.,7.])
        shifts = np.array([7.,3.,1.])
        sums = one_hot.one_hot_add(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals+shifts)%K).all(), "correct results")       
                
    def test_one_hot_minus(self):
        """Test one_hot_minus (if max value position moves by -shift)."""
        K = 8
        vals = np.array([1.,3.,7.])
        shifts = np.array([7.,3.,1.])
        sums = one_hot.one_hot_minus(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals-shifts)%K).all(), "correct results")       

    def test_one_hot_multiply(self):
        """Test one_hot_multiply (multiplying of one-hot encoded vectors)."""
        K = 8
        vals = np.array([1.,3.,7.])
        shifts = np.array([7.,3.,1.])
        sums = one_hot.one_hot_multiply(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals*shifts)%K).all(), "correct results")


    def test_one_hot_add_random(self):
        """Test one_hot_add (if max value position moves by shift)."""
        N, K = 1024, 123

        vals = np.random.choice(range(K),N)
        shifts = np.random.choice(range(K),N)
        sums = one_hot.one_hot_add(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals+shifts)%K).all(), "correct results")       
                
    def test_one_hot_minus_random(self):
        """Test one_hot_minus (if max value position moves by -shift)."""
        N, K = 1024, 123

        vals = np.random.choice(range(K),N)
        shifts = np.random.choice(range(K),N)
        sums = one_hot.one_hot_minus(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals-shifts)%K).all(), "correct results")       

    def test_one_hot_multiply_random(self):
        """Test one_hot_multiply (multiplying of one-hot encoded vectors)."""
        N, K = 1024, 123

        vals = np.random.choice(range(K),N)
        shifts = np.random.choice(range(1,K),N)
        sums = one_hot.one_hot_multiply(tf.one_hot(vals, K), tf.one_hot(shifts, K)) 

        # check if there's exactly one 1 per row and remaining are zeros:
        self.assertAllEqual( (tf.reduce_sum(sums,-1)), 1, "row sum==1")
        self.assertAllEqual( (tf.reduce_max(sums,-1)), 1, "max cell value in each row==1")
        self.assertAllEqual( (tf.reduce_min(sums,-1)), 0, "min cell value in each row==0")
        # check if results are correct
        self.assertTrue( (np.argmax(sums,-1)==(vals*shifts)%K).all(), "correct results")

if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    unittest.main()



