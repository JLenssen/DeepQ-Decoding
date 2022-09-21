# ------------ This script runs a training cycle for a single configuration point ---------------

from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.callbacks import FileLogger

from deepq.Function_Library import *
from deepq.Environments import *

import numpy as np
import tensorflow as tf

import pickle
import sys
import os
import datetime
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

RANDOM_SEED = fixed_configs["random_seed"]
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

memory_file = os.path.join(variable_configs_folder, "memory.p")
memory = pickle.load(open(memory_file, "rb"))

model = build_convolutional_nn(
    all_configs["c_layers"], all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(masked_greedy=all_configs["masked_greedy"]),
                              attr='eps', value_max=all_configs["max_eps"],
                              value_min=all_configs["final_eps"],
                              value_test=0.0,
                              nb_steps=all_configs["exploration_fraction"])
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

initial_weights_file = os.path.join(
    variable_configs_folder, "initial_dqn_weights.h5f")
dqn.model.load_weights(initial_weights_file)

# ---------------------------------------------------------------------------------------------

logging_path = os.path.join(variable_configs_folder, "training_history.json")
logging_callback = FileLogger(
    filepath=logging_path, interval=all_configs["print_freq"])
weights_path = os.path.join(
    variable_configs_folder, "final_dqn_weights.h5f")
memory_file = os.path.join(variable_configs_folder, "final_memory.p")
model_callback = CustomizedModelIntervalCheckpoint(filepath=weights_path, memorypath=memory_file,
                                         interval=all_configs["save_weight_freq"]
                                         )

# -------------------------------------------------------------------------------------------

now = datetime.datetime.now()
started_file = os.path.join(variable_configs_folder, "started_at.p")
pickle.dump(now, open(started_file, "wb"))

history = dqn.fit(env,
                  nb_steps=all_configs["max_timesteps"],
                  action_repetition=1,
                  callbacks=[logging_callback,model_callback],
                  verbose=2,
                  visualize=False,
                  nb_max_start_steps=0,
                  start_step_policy=None,
                  log_interval=all_configs["print_freq"],
                  nb_max_episode_steps=None,
                  episode_averaging_length=all_configs["rolling_average_length"],
                  success_threshold=all_configs["success_threshold"],
                  stopping_patience=all_configs["stopping_patience"],
                  min_nb_steps=all_configs["exploration_fraction"],
                  single_cycle=False)

# --------------------------------------------------------------------------------------------

pickle.dump(dqn.memory, open(memory_file, "wb"))
dqn.save_weights(weights_path, overwrite=True)
