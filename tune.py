from __future__ import division

import datetime
import errno
import os

import keras_tuner as kt
import tensorflow as tf
from keras import layers
from keras import regularizers
from tensorflow import keras

from network import BioRNN
from task import generate_trials
from train import get_default_hp


class MyHyperModel(kt.HyperModel):
    def build(self, hyperparams):
        cell = BioRNN(hyperparams.values['n_rnn'], hyperparams.values)  # Manually input seed
        x = keras.Input(shape=(None, hyperparams.values['n_input']))
        h = layers.RNN(cell,
                       activity_regularizer=regularizers.l1_l2(hyperparams['L1_activity'], hyperparams['L2_activity']),
                       time_major=False, return_sequences=True, name='RNN')(x)
        rnn_output = layers.Dense(hyperparams.values['n_output'], use_bias=False, name='output')(h)

        # Tune learning rate
        hp_learning_rate = hyperparams.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = tf.optimizers.Adam(learning_rate=hp_learning_rate, clipvalue=1.0)
        model = keras.Model(inputs=x, outputs=rnn_output, name='RNN_Model')
        model.compile(optimizer=optimizer, loss=hyperparams.values['loss_type'], sample_weight_mode="temporal")
        return model


class MyTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 12, 108, step=12)
        # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 1, 3, step=1)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)


def tune(seed_name, rule, seed, checkpoint_suffix='', hp=None, reload_model_name=None):
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
    #     fname = os.path.join(model_dir, 'history.json')
    #     with open(fname, 'r') as f:
    #         history = json.load(f)

    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    rule_train_now = rule
    # hp['seed'] = seed
    # Build and train model
    # cell = BioRNN(hp['n_rnn'], hp, seed)
    # model = build_model(hp, cell)

    # if reload_model_name is not None:
    #     model.load_weights(reload_model_name + checkpoint_suffix).expect_partial()

    # model.summary()

    trial = generate_trials(rule_train_now, hp, 'random')
    x, y, c_mask = trial.x, trial.y, trial.c_mask
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y, c_mask[:, :, 0]))
    # train_dataset = train_dataset.batch(hp['batch_size'])

    hyperparams = kt.HyperParameters()
    for key, value in hp.items():
        if key not in ['learning_rate', 'batch_size', 'epochs']:
            hyperparams.Fixed(key, value)

    # Uses same arguments as the BayesianOptimization Tuner.
    tuner = MyTuner(MyHyperModel(hyperparams),
                    objective='loss',
                    max_trials=10,
                    directory='tuning',
                    project_name='int_discrimTEST_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                    hyperparameters=hyperparams)

    tuner.search(train_dataset)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal batch size is {best_hps.get('batch_size')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}""")
