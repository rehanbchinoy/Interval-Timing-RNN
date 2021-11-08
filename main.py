"""MULTI.py performs multiple train calls across experiments, parameters"""

import os

import standard_analysis
import train
# from tune import tune
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

numExp = 1
SeedBase = 100

rule_list = ['Interval_Discrim']
checkpoint_suffix = '/checkpoint/'

# reload_model_name = 'MULTI_100_Interval_Discrim_parametric_anchor'
# reload_directory = 'models/' + reload_model_name


def run():
    for exp in range(numExp):

        seed = SeedBase + exp

        for rule in rule_list:
            # seed_name = reload_model_name + '_retrain'
            seed_name = 'MULTI_' + str(seed) + '_' + rule + '_' + 'bs_10'
            # train.train(seed_name, rule, seed, checkpoint_suffix, reload_directory=reload_directory)
            train.train(seed_name, rule, seed, checkpoint_suffix, hp={'batch_size': 10})
            # train.train(seed_name, rule, seed, checkpoint_suffix)

            # Visualize results and save to MATLAB
            # standard_analysis.summary_plots(seed_name, rule, seed, checkpoint_suffix)
            standard_analysis.write2mat(seed_name, rule, seed, checkpoint_suffix)

# Run Simulations--------------------------------------------------------
# run()

# Run tuner
# seed = 915
# seed_name = 'MULTI_910_Interval_Discrim_more_intervals_learning_rate0.0005'
# rule = 'Interval_Discrim'
# tune(seed_name, rule, seed, checkpoint_suffix='', hp=None, reload_model_name=None)
# Write to MATLAB--------------------------------------------------------
# model_ind = 907
# num_exp = 1
# i = 0
rule = 'Interval_Discrim'
# # hp = get_default_hp()
# # while i < num_exp:
# #     model_dir = 'MULTI_' + str(model_ind + i) + '_' + rule + '_' + 'randomize_delay_1'
# #     for j in range(5):
# #         hp['delay'] = 250*(j+1)
# #         standard_analysis.write2mat(model_dir, rule, checkpoint_suffix, hp)
# #     i+=1
#
#
model_dir = 'MULTI_100_Interval_Discrim_parametric_anchor'
# if model_dir + "/hp.txt":
#     file = open("models/" + model_dir + "/hp.txt", "r")
#     contents = file.read()
#     hp = ast.literal_eval(contents)
#     file.close()
#     print(hp)
# hp['delay'] = 500
standard_analysis.write2mat(model_dir, rule, 100, checkpoint_suffix)
