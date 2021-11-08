"""Standard analyses that can be performed on any task"""

from __future__ import division

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from train import get_model_parameters


def write2mat(seed_name, rule, seed, checkpoint_suffix=''):
    model_dir = 'models/' + seed_name
    # fname = os.path.join(model_dir, 'hp.json')
    # with open(fname, 'r') as f:
    #     hp = json.load(f)
    # params = get_model_parameters(model_dir=model_dir, rule=rule, seed=seed, checkpoint_suffix=checkpoint_suffix, hp=hp)
    params = get_model_parameters(model_dir=model_dir, rule=rule, seed=seed, checkpoint_suffix=checkpoint_suffix)
    scipy.io.savemat(model_dir + '/' + seed_name + '_untrained_intervals' + '.mat', params)


def summary_plots(seed_name, rule, seed, checkpoint_suffix=''):
    model_dir = 'models/' + seed_name
    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'r') as f:
        hp = json.load(f)
    params = get_model_parameters(model_dir=model_dir, rule=rule, seed=seed, checkpoint_suffix=checkpoint_suffix, hp=hp)

    # Plot sensory input, motor output, motor target, cost mask
    for i in range(5):
        plt.figure()
        if rule == 'Dur_Discrim' or rule == 'Interval_Discrim':
            if params['respond'][i]:
                # if params['stim_order'][i]:
                #     plt.title('Diff. dur. (Std. second) --> response')
                # else:
                plt.title('First < Second --> response')
            else:
                plt.title('First > Second --> no response')

        plt.plot(params['y'][i, :, 0], label='Target')
        # plt.plot(np.mean(params['x'][i, :, :], 1), label='Input')
        # plt.plot(params['y_hat'][i, :, 0], label='Out')
        if rule == 'ITdDNMS':
            plt.plot(params['y'][i, :, 1], label='Attention Target')
            # plt.plot(params['y_hat'][i, :, 1], label='Attention')
        plt.ylim([-0.5, 2.5])
        plt.plot(params['c_mask'][i, :, 0], label='Loss Mask')
        plt.legend(loc='upper center', shadow=True, fontsize='small')
        plt.show()

        plt.figure()
        plt.plot(params['x'][i, :, :])
        plt.show()

    # Plot weights
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.imshow(params['w_rec'])
    ax2 = fig.add_subplot(132)
    ax2.imshow(params['w_in'])
    ax3 = fig.add_subplot(133)
    ax3.imshow(params['w_out'])
    plt.show()

    # Plot recurrent activity
    h = np.squeeze(np.transpose([params['h'][1, :, :]]))
    ind_peak = np.argmax(h, 1)
    ind_sort = np.argsort(ind_peak)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    im = ax1.imshow(h[ind_sort, :])
    fig.colorbar(im)
    ax2 = fig.add_subplot(212)
    ax2.plot(h[:, :].T)
    plt.show()
