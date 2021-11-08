"""Main training loop"""

from __future__ import division

import datetime
import errno
import json
import math
import os
import time
import warnings

import numpy as np
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras.callbacks import TensorBoard
from tensorflow import keras

from network import BioRNN
from task import generate_trials


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size, x, y, c_mask, hp, rule, *args, **kwargs):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.c_mask = c_mask
        self.hp = hp
        self.rule = rule
        self.x_noise = self.add_noise()

    def __len__(self):
        # returns the number of batches
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        # returns one batch
        ind = np.arange(index * self.batch_size,(index + 1) * self.batch_size, 1)
        x_batch = self.x_noise[ind, :, :]
        y_batch = self.y[ind, :, :]
        c_mask_batch = self.c_mask[ind, :, :]
        return x_batch, y_batch, c_mask_batch

    def add_noise(self):
        x_noise = self.x + np.random.RandomState().randn(*self.x.shape) * self.hp['sigma_x']
        return x_noise

    def on_epoch_end(self):
        # shuffle and add noise for "new" dataset
        # seed = time.time()
        # idx = tf.random.shuffle(tf.range(int(self.hp['dataset_size'])))

        # shuffled_indices = tf.random.shuffle(indices)

        # self.x = tf.gather(self.x, idx)
        # self.y = tf.gather(self.y, idx)
        # self.c_mask = tf.gather(self.c_mask, idx)

        # tf.random.shuffle(self.x, seed)
        # tf.random.shuffle(self.y, seed)
        # tf.random.shuffle(self.c_mask, seed)
        # self.x_noise = self.add_noise()
        # print(self.x_noise[1,1,0])

        trial = generate_trials('Interval_Discrim', self.hp, 'random', noise_on=True)
        self.x_noise, self.y, self.c_mask = trial.x, trial.y, trial.c_mask

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def get_default_hp():
    hp = {
        'activation': 'relu',
        'alpha': 0.2,  # originally 0.2
        'batch_size': 10,
        'bias_rec_init': 0,
        'Dales_law': True,  # False=+/- weights; True=Ex/Inh neurons
        'dataset_size': 400,  # includes both training and validation
        'delay': 500,
        'dt': 10,
        'ei_cell_ratio': 0.8,  # 1 = 100% Ex; 0.75=80%/20% #originally 0.8
        'epochs': 4000,
        'epoch_size': 500, #TODO: consider removing
        'learning_rate': 0.001,
        'loss_type': 'mse',
        'L1_activity': 0,
        'L2_activity': 0.000001,
        'L1_weight': 0,
        'L2_weight': 0,
        'n_input': 1,
        'n_output': 1,
        'n_rnn': 256,
        'optimizer': 'Adam',
        'sigma_rec': 0,  # recurrent noise #DNMS: .05 Dean's: 0.005
        'sigma_x': 0.005,  # DNMS: .01 Dean's: 0.005
        'std_dur': 200,
        'tone_dur': 20,  # 2*dt
        'target_loss': 0.0005,  # TODO: implement this
        'train_bias_rec': False,
        'train_w_in': False,
        'validation_split': 0,
        'w_rec_gain': 0.1,  # initial w0_rec_init gain # originally 0.1
        'w_rec_init': 'randortho',  # randortho, randgauss, diag
    }
    return hp

def get_model_parameters(model_dir, rule, seed, checkpoint_suffix='', hp=None):
    ckpt_dir = model_dir + checkpoint_suffix

    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'r') as f:
        hp = json.load(f)

    fname = os.path.join(model_dir, 'times.json')
    with open(fname, 'r') as f:
        times = json.load(f)

    fname = os.path.join(model_dir, 'loss.json')
    with open(fname, 'r') as f:
        loss = json.load(f)

    if hp == None:
        hp = get_default_hp()

    if hp['seed'] == None:
        hp['seed'] = seed

    # hp['sigma_rec'] = 0.01

    cell = BioRNN(hp['n_rnn'], hp)
    model = build_model(hp, cell)
    model.load_weights(ckpt_dir).expect_partial()  # Load model with least loss for visualizing weights
    trial = generate_trials(rule, hp, 'random')
    w_rec_init, EI_matrix, autapse_mask = cell.w_rec0, cell.ei_cells, cell.autapse_mask
    x, y, c_mask, respond, std_order, delay, std_dur, comp_dur, int1_ons, int1_offs, int2_ons, \
    int2_offs = trial.x, trial.y, trial.c_mask, trial.respond, trial.std_order, trial.delay, trial.std_dur, \
                trial.comp_dur, trial.int1_ons, trial.int1_offs, trial.int2_ons, trial.int2_offs
    y_hat = model.predict(x)
    l_rnn = model.get_layer('RNN')
    w_rnn = l_rnn.get_weights()
    wgt_layer = model.get_layer('output')
    w_out = wgt_layer.get_weights()
    h = l_rnn(x).numpy()
    if hp['train_bias_rec']:
        b_rec = w_rnn[1]
        w_in = w_rnn[2]
    else:
        b_rec = w_rnn[2]
        w_in = w_rnn[1]
    params = {'w_rec_init': w_rec_init, 'EI_matrix': EI_matrix, 'autapse_mask': autapse_mask,
              'std_order': std_order, 'w_in': w_in, 'w_rec': w_rnn[0],
              'w_out': w_out[0], 'x': x, 'y': y, 'y_hat': y_hat, 'h': h, 'c_mask': c_mask,
              'b_rec': b_rec, 'respond': respond, 'loss': loss, 'times': times, 'hp': hp, 'delay': delay,
              'std_dur': std_dur, 'comp_dur': comp_dur, 'int1_ons': int1_ons, 'int1_offs': int1_offs,
              'int2_ons': int2_ons, 'int2_offs': int2_offs}

        # params = {'w_rec_init': w_rec_init, 'EI_matrix': EI_matrix, 'autapse_mask': autapse_mask,
        #           'std_order': std_order, 'w_in': w_in, 'w_rec': w_rnn[0],
        #           'w_out': w_out[0], 'x': x, 'y': y, 'y_hat': y_hat, 'h': h, 'c_mask': c_mask,
        #           'b_rec': b_rec, 'respond': respond, 'hp': hp, 'delay': delay,
        #           'std_dur': std_dur, 'comp_dur': comp_dur, 'int1_ons': int1_ons, 'int1_offs': int1_offs,
        #           'int2_ons': int2_ons, 'int2_offs': int2_offs}
    return params

def build_model(hp, cell):
    # Build model using the Keras Functional API with custom Bio_RNN cells
    x = keras.Input(shape=(None, hp['n_input']))
    h = layers.RNN(cell, activity_regularizer=regularizers.l1_l2(hp['L1_activity'], hp['L2_activity']),
                   time_major=False, return_sequences=True, name='RNN')(x)
    rnn_output = layers.Dense(hp['n_output'], use_bias=False, name='output')(h)
    optimizer = tf.optimizers.Adam(learning_rate=hp['learning_rate'], clipvalue=1.0)
    loss = tf.keras.losses.MeanSquaredError()
    model = tf.keras.Model(inputs=x, outputs=rnn_output, name='RNN_Model')
    model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    return model

# @tf.function
# def train_step(x, y, c_mask, model):
#     with tf.GradientTape() as tape:
#         y_pred = model(x, training=True)
#         loss_value = model.loss(y, y_pred, c_mask)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     return loss_value

def train(seed_name, rule, seed, checkpoint_suffix='', hp=None, reload_directory=None):
    # Make model directory (allow overwriting)
    model_dir = 'models/' + seed_name
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(model_dir):
            pass
        else:
            raise

    # Set training parameters
    # if reload_model_name is not None:
    #     fname = os.path.join(reload_model_name, 'hp.json')
    #     with open(fname, 'r') as f:
    #         hp = json.load(f)
    if reload_directory is not None:
        fname = os.path.join(reload_directory, 'hp.json')
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        default_hp = get_default_hp()
        if hp is not None:
            default_hp.update(hp)
        hp = default_hp
        hp['rule'] = rule
        hp['seed'] = seed

    # Build and train model
    cell = BioRNN(hp['n_rnn'], hp)
    model = build_model(hp, cell)
    if reload_directory is not None:
         model.load_weights(reload_directory + checkpoint_suffix).expect_partial()
    model.summary()

    tensorboard_dir = "tensorboard_logs/" + seed_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = model_dir + checkpoint_suffix
    time_callback = TimeHistory()
    early_stopping = EarlyStoppingByLossVal(monitor='loss', value=hp['target_loss'], verbose=3)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1, profile_batch='2,10')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_dir, save_weights_only=True,
        monitor='loss', mode='min', save_best_only=True, verbose=3)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

###############################################################
    #BENCHMARKS

    ## DATASET GENERATOR METHOD
    trial = generate_trials('Interval_Discrim', hp, 'random', noise_on=True)
    x, y, c_mask = trial.x, trial.y, trial.c_mask
    dataset_gen = CustomDataset(hp['batch_size'], x, y, c_mask, hp, rule)
    history = model.fit(dataset_gen, epochs=hp['epochs'], steps_per_epoch=int(hp['dataset_size']/hp['batch_size']),
                        callbacks=[time_callback, early_stopping, checkpoint_callback])
    times = {'times': time_callback.times}
    loss = history.history

    ### MAP FUNCTION
    # history = model.fit(dataset.repeat(hp['epochs']).shuffle(hp['dataset_size']).map(lambda func, inp, Tout: tf.py_function(func=add_noise, inp=[dataset], Tout=(tf.float32, tf.float32, tf.float32))).batch(hp['batch_size']),
    # epochs=hp['epochs'], steps_per_epoch=int(hp['dataset_size']/hp['batch_size']),
    #                     callbacks=[time_callback, early_stopping, checkpoint_callback])

    ### REPEATED DATASET
    # history = model.fit(dataset.repeat(np.ceil(hp['epochs']*hp['epoch_size']/hp['dataset_size'])).shuffle(hp['epoch_size']).batch(25),
    #                     epochs=hp['epochs'], steps_per_epoch=int(hp['epoch_size']/25), callbacks=[time_callback, early_stopping, checkpoint_callback])


    # Model fit loop generating every epoch
    # loss = []
    # times = []
    # for epochs in range(hp['epochs']):
    #     epoch_start_time = time.time()
    #     trial = generate_trials(rule, hp, 'random', noise_on=True)
    #     x, y, c_mask = trial.x, trial.y, trial.c_mask
    #     dataset = tf.data.Dataset.from_tensor_slices((x, y, c_mask[:, :, 0]))
    #     dataset = dataset.batch(hp['batch_size'])
    #
    #     for step, (x, y, c_mask) in enumerate(dataset):
    #         loss_value = train_step(x, y, c_mask, model)
    #         loss += loss_value
    #         # times += (time.time() - epoch_start_time)
    #         if epochs % 1 == 0 and step % (int(hp['dataset_size']/hp['batch_size'])) == 0:
    #             print("Training loss at epoch %d (for one batch) at step %d: %.4f"
    #             % (epochs, step, float(loss_value)))
        # history = model.fit(dataset, epochs=1, verbose=2, steps_per_epoch=hp['dataset_size'] / hp['batch_size'],
        #         callbacks=[time_callback, early_stopping, checkpoint_callback])
        # loss += history.history['loss']
        # times += time_callback.times

    # Model fit loop generating at start of training only
    # loss = []
    # times = []
    # trial = generate_trials('Interval_Discrim', hp, 'random', noise_on=True)
    # x, y, c_mask = trial.x, trial.y, trial.c_mask
    # dataset = tf.data.Dataset.from_tensor_slices((x, y, c_mask[:, :, 0]))
    # dataset = dataset.batch(hp['batch_size'])
    # for epochs in range(hp['epochs']):
    #     # for step, (x, y, c_mask) in enumerate(dataset):
    #     #     loss_value = train_step(x, y, c_mask, model)
    #     #     loss += loss_value
    #     #     times[] += (time.time() - start_time)
    #     #     if epochs % 5 == 0 and step % hp['batch_size'] == 0:
    #     #         print("Training loss at epoch %d (for one batch) at step %d: %.4f"
    #     #         % (epochs, step, float(loss_value))
    #     #         )
    #     #         print("Time taken: %.2fs" % (time.time() - start_time))
    #     history = model.fit(dataset, epochs=1, verbose=2, steps_per_epoch=hp['dataset_size'] / hp['batch_size'],
    #                         callbacks=[time_callback, early_stopping, checkpoint_callback])
    #     loss += history.history['loss']
    #     times += time_callback.times


    # Write training data to json files
    fname = os.path.join(model_dir, 'times.json')
    with open(fname, 'w') as f:
        json.dump(times, f)

    fname = os.path.join(model_dir, 'loss.json')
    with open(fname, 'w') as f:
        json.dump(loss, f)

    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'w') as f:
        json.dump(hp, f)
