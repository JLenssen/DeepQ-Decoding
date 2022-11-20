# ------------ This script runs a training cycle for a single configuration point ---------------

from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

import numpy as np
import tensorflow as tf

from deepq.Function_Library import *
from deepq.Environments import *

import pickle
import sys
import os
import random

# ---------------------------------------------------------------------------------------------

variable_config_number = sys.argv[1]
base_directory = sys.argv[2]

variable_configs_folder = os.path.join(
    base_directory, "config_"+str(variable_config_number) + "/")
variable_configs_path = os.path.join(
    variable_configs_folder, "variable_config_"+variable_config_number + ".p")
fixed_configs_path = os.path.join(os.path.dirname(base_directory), "fixed_config.p")

fixed_configs = pickle.load(open(fixed_configs_path, "rb"))
variable_configs = pickle.load(open(variable_configs_path, "rb"))

all_configs = {}

for key in fixed_configs.keys():
    all_configs[key] = fixed_configs[key]

for key in variable_configs.keys():
    all_configs[key] = variable_configs[key]

if fixed_configs["static_decoder"]:
  static_decoder = load_model(os.path.join(
      base_directory, "static_decoder"))
else:
  static_decoder = None

# -------------------------------------------------------------------------------------------

RANDOM_SEED = all_configs["random_seed"]
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

# --------------------------------------------------------------------------------------------

noise_model = NoiseFactory(
    all_configs["error_model"], all_configs["d"], all_configs["p_phys"]).generate()

env = Surface_Code_Environment_Multi_Decoding_Cycles(d=all_configs["d"],
                                                     p_meas=all_configs["p_meas"],
                                                     noise_model=noise_model,
                                                     use_Y=all_configs["use_Y"],
                                                     volume_depth=all_configs["volume_depth"],
                                                     static_decoder=static_decoder)

# -------------------------------------------------------------------------------------------

final_weights_file = os.path.join(
    variable_configs_folder, "final_dqn_weights.h5f")

# -------------------------------------------------------------------------------------------

model = build_convolutional_nn(
    all_configs["c_layers"], all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
memory = SequentialMemory(limit=all_configs["buffer_size"], window_length=1)
policy = GreedyQPolicy(masked_greedy=True)
test_policy = GreedyQPolicy(masked_greedy=True)

# ------------------------------------------------------------------------------------------

dqn = DQNAgent(model=model,
               nb_actions=env.num_actions,
               memory=memory,
               nb_steps_warmup=all_configs["learning_starts"],
               target_model_update=all_configs["target_network_update_freq"],
               policy=policy,
               test_policy=test_policy,
               gamma=all_configs["gamma"],
               enable_dueling_network=all_configs["dueling"])


dqn.compile(Adam(lr=all_configs["learning_rate"]))
dqn.model.load_weights(final_weights_file)

# -------------------------------------------------------------------------------------------

trained_at = all_configs["p_phys"]
num_to_test = 20
error_rates = [j*0.001 for j in range(1, num_to_test + 1)]
thresholds = [1/p for p in error_rates]
nb_test_episodes = all_configs["testing_length"]
all_results = {}


keep_evaluating = True
count = 0
while keep_evaluating:

  err_rate = error_rates[count]

  print(f"Evaluating error rate: {err_rate}")
  noise_model = NoiseFactory(
    all_configs["error_model"], all_configs["d"], err_rate).generate()
  env = Surface_Code_Environment_Multi_Decoding_Cycles(d=all_configs["d"],
                                                      p_meas=err_rate,
                                                      noise_model=noise_model,
                                                      use_Y=all_configs["use_Y"],
                                                      volume_depth=all_configs["volume_depth"],
                                                      static_decoder=static_decoder)

  dict_key = str(err_rate)[:5]

  testing_history = dqn.test(env, nb_episodes=nb_test_episodes,
                             visualize=False, verbose=2, interval=10, single_cycle=False)
  results = testing_history.history["episode_lifetimes_rolling_avg"]
  final_result = results[-1:][0]
  all_results[dict_key] = testing_history.history

  if abs(trained_at - err_rate) < 1e-6:
    results_file = os.path.join(variable_configs_folder, "results.p")
    pickle.dump(results, open(results_file, "wb"))

  count += 1
  if count == num_to_test:
    keep_evaluating = False

all_results_file = os.path.join(variable_configs_folder, "all_results.p")
pickle.dump(all_results, open(all_results_file, "wb"))
