#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from utils import *
import pdb
from time import gmtime, strftime

from config import Config

class SeparationModel():
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        TODO: Add these placeholders to self as the instance variables
            self.inputs_placeholder
            self.targets_placeholder
            self.seq_lens_placeholder

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op. 
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features]. 

        (Don't change the variable names)
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features), name='inputs')
        self.targets_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.output_size), name='targets')

    def create_feed_dict(self, inputs_batch, targets_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """        
        feed_dict = {
            self.inputs_placeholder: [tf.transpose(inputs_batch)],
            self.targets_placeholder: [tf.transpose(targets_batch)],
        }

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete 
        in this function: 

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden]. 
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This 
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to 
          "logits". 

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """

        gru_cell = tf.contrib.rnn.GRUCell(Config.output_size, input_size=Config.num_final_features, activation=tf.nn.relu)

        if Config.num_layers > 1:
            # multi layer
            a = 1

        output, state = tf.nn.dynamic_rnn(gru_cell, self.inputs_placeholder, dtype=tf.float32)

        self.output = output


    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph. 

        - Use tf.nn.ctc_loss to calculate the CTC loss for each example in the batch. You'll need self.logits,
          self.targets_placeholder, self.seq_lens_placeholder for this. Set variable ctc_loss to
          the output of tf.nn.ctc_loss
        - You will need to first tf.transpose the data so that self.logits is shaped [max_timesteps, batch_s, 
          num_classes]. 
        - Configure tf.nn.ctc_loss so that identical consecutive labels are allowed
        - Compute L2 regularization cost for all trainable variables. Use tf.nn.l2_loss(var). 

        """
        l2_cost = 0.0

        squared_error = tf.norm(self.output - self.targets_placeholder, ord=2)
        self.loss = Config.l2_lambda * l2_cost + squared_error

        tf.summary.scalar("squared_error", squared_error)       
        
        tf.summary.scalar("loss", self.loss)

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model. Call optimizer.minimize() on self.loss. 

        """
        optimizer = None 

        ### YOUR CODE HERE (~1-2 lines)
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)
        ### END YOUR CODE
        
        self.optimizer = optimizer

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    # This actually builds the computational graph 
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
        self.add_summary_op()
        

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        batch_cost, summary = session.run([self.loss, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, summary

    def print_results(self, train_inputs_batch, train_targets_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)        

    def __init__(self):
        self.build()

    

