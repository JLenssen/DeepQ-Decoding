# ------------ This script runs a training cycle for a single configuration point ---------------

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

import numpy as np
import tensorflow as tf

from deepq.Function_Library import *
from deepq.Environments import *

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

RANDOM_SEED = all_configs["random_seed"]
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

NUM_PARALLEL_EXEC_UNITS=1
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=0, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
session = tf.Session(config=config)
K.set_session(session)

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

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

model = build_convolutional_nn(
    all_configs["c_layers"], all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
memory = SequentialMemory(limit=all_configs["buffer_size"], window_length=1)
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

# ---------------------------------------------------------------------------------------------

callbacks = []
if fixed_configs["use_tensorboard"]:
  tensorboard_logging_path = os.path.join(
      base_directory, "tensorboard_logs", "config_"+str(variable_config_number))
  # Choosing update frequency to be epoch which means that TensorBoard will update after each episode.
  # TensorBoard is only used to find suitable hyper-parameters and is not needed in actual training.
  callbacks.append(TensorBoard(log_dir=tensorboard_logging_path,
                   histogram_freq=0, update_freq='epoch'))

logging_path = os.path.join(variable_configs_folder, "training_history.json")
callbacks.append(FileLogger(filepath=logging_path,
                 interval=all_configs["print_freq"]))
weights_path = os.path.join(
    variable_configs_folder, "final_dqn_weights.h5f")
memory_file = os.path.join(variable_configs_folder, "final_memory.p")
callbacks.append(CustomizedModelIntervalCheckpoint(filepath=weights_path, memorypath=memory_file,
                                         interval=all_configs["save_weight_freq"]
                                         ))

# -------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------

now = datetime.datetime.now()
started_file = os.path.join(variable_configs_folder, "started_at.p")
pickle.dump(now, open(started_file, "wb"))

history = dqn.fit(env,
                  nb_steps=all_configs["max_timesteps"],
                  action_repetition=1,
                  callbacks=callbacks,
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

# -------------------------------------------------------------------------------------------

pickle.dump(dqn.memory, open(memory_file, "wb"))
dqn.save_weights(weights_path, overwrite=True)
