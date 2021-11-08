"""Definition of the network model and various RNN cells"""

from __future__ import division

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BioRNN(tf.keras.layers.AbstractRNNCell):
    """The most basic RNN cell with external inputs separated.

    Args:
        num_units: int, The number of units in the RNN cell.
    """

    def __init__(self, num_units, hp):  # TODO: Why does num_units need to be its own arg
        super(BioRNN, self).__init__()
        self.units = num_units
        self.hp = hp
        self._num_units = num_units
        self.seed = self.hp['seed']
        self._w_rec_init = self.hp['w_rec_init']
        self._bias_rec_init = self.hp['bias_rec_init']
        self.Dales_law = self.hp['Dales_law']
        self._ei_cell_ratio = self.hp['ei_cell_ratio']  # not used if Dales_law = False
        self._activation = keras.activations.get(self.hp['activation'])
        self._alpha = self.hp['alpha']
        self._sigma = np.sqrt(2 / self.hp['alpha']) * self.hp['sigma_rec']
        self.rng = np.random.RandomState(self.seed)
        self.L1_weight = self.hp['L1_weight']
        self.L2_weight = self.hp['L2_weight']

        n_rec = self.units
        # Dales Law
        if self.Dales_law:
            self.nEx = int(n_rec * self._ei_cell_ratio)
            self.nInh = n_rec - self.nEx
            _ei_list = np.ones(n_rec)
            _ei_list[-self.nInh:] = -1
            self.ei_cells = np.diag(_ei_list)
            self.ei_cells = self.ei_cells.astype(np.float32)
        else:
            self.ei_cells = np.eye(n_rec)
            self.ei_cells = self.ei_cells.astype(np.float32)

        # Generate w0_rec initialization matrix
        if self._w_rec_init == 'diag':
            w_rec0 = self.hp['w_rec_gain'] * np.eye(n_rec)
        elif self._w_rec_init == 'randortho':
            initializer = tf.keras.initializers.Orthogonal()
            values = initializer(shape=(n_rec, n_rec))
            w_rec0 = self.hp['w_rec_gain'] * values.numpy()
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self.hp['w_rec_gain'] * self.rng.randn(n_rec, n_rec) / np.sqrt(n_rec))
        else:
            raise ValueError

        if self.Dales_law:
            w_rec0 = np.abs(w_rec0)
            w_rec0[-self.nInh:, :] = w_rec0[-self.nInh:, :] * self._ei_cell_ratio / (1 - self._ei_cell_ratio)

        # Zero Diagonal = no autapses
        w_rec0[range(n_rec), range(n_rec)] = 0
        self.w_rec0 = w_rec0
        self.recurrent_kernel_initializer = tf.constant_initializer(w_rec0)

        # autapse Mask => keep autapses = 0
        autapse_mask = np.ones((n_rec, n_rec))
        autapse_mask[range(n_rec), range(n_rec)] = 0
        self.autapse_mask = autapse_mask.astype(np.float32)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self._num_units

    def get_config(self):

        config = super(BioRNN, self).get_config()
        config.update({
            'num_units': self.units,
            'hp': self.hp,
            'w_rec_init': self._w_rec_init,
            'activation': self._activation,
            # 'w_in_start': self._w_in_start,
            # 'w_rec_start': self._w_rec_start,
            'alpha': self._alpha,
            'sigma': self._sigma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, inputs_shape):
        self.w_in_initializer = tf.random_normal_initializer(0, stddev=1 / np.sqrt(self.hp['n_input']))

        self.w_in = self.add_weight(shape=(inputs_shape[-1], self.units),
                                    initializer=self.w_in_initializer,
                                    trainable=self.hp['train_w_in'], name='w_in')

        if self.Dales_law:
            self.w_rec = self.add_weight(shape=[self.units, self.units],
                                         initializer=self.recurrent_kernel_initializer, name='w_rec',
                                         constraint=tf.keras.constraints.NonNeg(),
                                         regularizer=keras.regularizers.l1_l2(self.L1_weight, self.L2_weight))

        else:
            self.w_rec = self.add_weight(shape=[self.units, self.units],
                                         initializer=self.recurrent_kernel_initializer,
                                         regularizer=keras.regularizers.l1_l2(self.L1_weight, self.L2_weight),
                                         name='w_rec')

        self.b_rec = self.add_weight(shape=[self.units],
                                     initializer=tf.keras.initializers.Constant(self.hp['bias_rec_init']),
                                     trainable=self.hp['train_bias_rec'], name='b_rec')

        self.built = True

    def call(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""
        inp = tf.linalg.matmul(inputs, self.w_in)
        inp = inp + tf.linalg.matmul(tf.linalg.matmul(state, self.ei_cells), self.w_rec * self.autapse_mask)
        inp = tf.nn.bias_add(inp, self.b_rec)

        noise = tf.random.normal(tf.shape(input=state), mean=0, stddev=self._sigma)
        inp = inp + noise

        output = self._activation(inp)
        output = tf.math.multiply((1 - self._alpha), state) + self._alpha * output
        output = tf.squeeze(output)
        # return output, [output]  # i.e. (output, new_state); output is the same as the next hidden state
        return output, output
