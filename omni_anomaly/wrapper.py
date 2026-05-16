# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense, RNN, GRUCell
import tensorflow_probability as tfp
from tfsnippet.distributions import Distribution


class TfpDistribution(Distribution):
    """
    A wrapper class for `tfp.distributions.Distribution`
    """

    @property
    def is_continuous(self):
        return self._is_continuous

    def __init__(self, distribution):
        if not isinstance(distribution, tfp.distributions.Distribution):
            raise TypeError('`distribution` is not an instance of `tfp.'
                            'distributions.Distribution`')
        super(TfpDistribution, self).__init__()
        self._distribution = distribution
        self._is_continuous = True
        self._is_reparameterized = self._distribution.reparameterization_type is tfp.distributions.FULLY_REPARAMETERIZED

    def __repr__(self):
        return 'Distribution({!r})'.format(self._distribution)

    @property
    def dtype(self):
        return self._distribution.dtype

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def value_shape(self):
        return self._distribution.event_shape

    def get_value_shape(self):
        return self._distribution.event_shape

    @property
    def batch_shape(self):
        return self._distribution.batch_shape

    def get_batch_shape(self):
        return self._distribution.batch_shape()

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False,
               name=None):
        from tfsnippet.stochastic import StochasticTensor
        if n_samples is None or n_samples < 2:
            n_samples = 2
        with tf.compat.v1.name_scope(name=name, default_name='sample'):
            samples = self._distribution.sample(n_samples)
            samples = tf.reduce_mean(samples, axis=0)
            t = StochasticTensor(
                distribution=self,
                tensor=samples,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=self.is_reparameterized
            )
            if compute_density:
                with tf.compat.v1.name_scope('compute_prob_and_log_prob'):
                    log_p = t.log_prob()
                    t._self_prob = tf.exp(log_p)
            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.compat.v1.name_scope(name=name, default_name='log_prob'):
            log_prob, _, _, _, _, _, _ = self._distribution.forward_filter(given)
            return log_prob


def softplus_std(inputs, layers, epsilon):
    return tf.nn.softplus(apply_dense(inputs, layers)) + epsilon

    
def apply_dense(inputs, dense_layer):
    return dense_layer(inputs)


def rnn(x,
        window_length,
        rnn_num_hidden,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=200,
        time_axis=1,
        name='rnn'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        if len(x.shape) == 4:
            x = tf.reduce_mean(x, axis=0)
        elif len(x.shape) != 3:
            logging.error("rnn input shape error")
        #x = tf.unstack(x, window_length, time_axis)

        if rnn_cell == 'LSTM':
            # Define lstm cells with TensorFlow
            # Forward direction cell
            fw_cell = tf.keras.layers.LSTMCell(rnn_num_hidden, forget_bias=1.0)
        elif rnn_cell == "GRU":
            fw_cell = tf.keras.layers.GRUCell(rnn_num_hidden)
        elif rnn_cell == 'Basic':
            fw_cell = tf.keras.layers.SimpleRNNCell(rnn_num_hidden)
        else:
            raise ValueError("rnn_cell must be LSTM or GRU")

        # Get lstm cell output

        try:
            outputs, _ = tf.keras.layers.RNN(fw_cell, unroll=True)(x)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.keras.layers.RNN(fw_cell, return_sequences=True, unroll=True)(x)
        #outputs = tf.stack(outputs, axis=time_axis)
        for i in range(hidden_dense):
            outputs = tf.keras.layers.Dense(dense_dim)(outputs)
        return outputs
    # return size: (batch_size, window_length, rnn_num_hidden)


def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    with tf.compat.v1.variable_scope('hidden', reuse=tf.compat.v1.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'mean': mean_layer()(h),
        'std': std_layer(h),
    }


def wrap_params_net_srnn(inputs, h_for_dist):
    with tf.compat.v1.variable_scope('hidden', reuse=tf.compat.v1.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'input_q': h
    }
