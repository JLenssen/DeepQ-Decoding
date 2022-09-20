import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

from deepq.Function_Library import *
from deepq.Environments import *

import pickle
import sys
import os
import random
import pprint

# ---------------------------------------------------------------------------------------------

RANDOM_SEED = 0
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

# --- Model evaluation ---------------------------------------------------------------------------

def evaluate_model(model_configs):
    # --- Build env ---------------------------------------------------------------------------------

    noise_model = XNoise(model_configs["d"], model_configs["p_phys"]/3, context='DP')

    env = Surface_Code_Environment_Multi_Decoding_Cycles(
        d=model_configs["d"],
        p_meas=model_configs["p_meas"],
        noise_model=noise_model,
        use_Y=model_configs["use_Y"],
        volume_depth=model_configs["volume_depth"],
        static_decoder=None,
    )

    # --- Build model --------------------------------------------------------------------------------

    dqn = build_eval_agent_model(model_configs, env)

    # --- Model evaluation ---------------------------------------------------------------------------

    trained_at = model_configs["p_phys"]
    num_to_test = 20
    error_rates = [j*0.001 for j in range(1, num_to_test + 1)]
    thresholds = [1/p for p in error_rates]
    nb_test_episodes = model_configs["testing_length"]
    all_results = {}


    keep_evaluating = True
    count = 0
    while keep_evaluating:
        err_rate = error_rates[count]
        env.noise_model.p_phys = err_rate
        env.p_meas = err_rate

        dict_key = str(err_rate)[:5]

        testing_history = dqn.test(env, nb_episodes=nb_test_episodes,
                                visualize=False, verbose=2, interval=10, single_cycle=False)
        results = testing_history.history["episode_lifetimes_rolling_avg"]
        final_result = results[-1:][0]
        all_results[dict_key] = final_result

        to_beat = thresholds[count]
        if final_result < to_beat or count == (num_to_test - 1):
            keep_evaluating = False

        count += 1

    os.mkdir(os.path.join(output_dir, str(trained_at)))
    all_results_file = os.path.join(output_dir, str(trained_at), "all_results.p")
    pickle.dump(all_results, open(all_results_file, "wb"))

# -----------------------------------------------------------------------------------------------

def load_agent_config(trained_model_dir: str, error_rate: str):
    fixed_configs_path = os.path.join(trained_model_dir, "fixed_config.p")
    base_config_path = os.path.join(trained_model_dir, error_rate)
    # find config variable
    for file in os.listdir(os.path.join(base_config_path)):
        if file.startswith("variable_config"):
            variable_configs_path = os.path.join(base_config_path, file)
            break
    model_weights_path = os.path.join(
        base_config_path, "final_dqn_weights.h5f")

    fixed_configs = pickle.load(open(fixed_configs_path, "rb"))
    variable_configs = pickle.load(open(variable_configs_path, "rb"))

    all_configs = {}
    for key in fixed_configs.keys():
        all_configs[key] = fixed_configs[key]
    for key in variable_configs.keys():
        all_configs[key] = variable_configs[key]
    all_configs["model_weights_path"] = model_weights_path
    return all_configs

# -----------------------------------------------------------------------------------------------

def run_eval_loop(training_dir, error_rate):
    model_configs = load_agent_config(training_dir, error_rate)
    print(f"Evaluating model: {error_rate}")
    pprint.pprint(model_configs)
    evaluate_model(model_configs)


# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":

    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        print("Usage: python3 src/robustness_evaluation.py <training_dir> <outputdir> [<error_rate>]")
        sys.exit(1)

    training_dir = sys.argv[1] # path to trained model configs
    output_dir = sys.argv[2] # output directory to store pickle files of test results
    if len(sys.argv) != 4:
        error_rate = sys.argv[3]
        run_eval_loop(training_dir, error_rate)
    else:
        for error_rate in os.listdir(training_dir): # loop over all configs trained at different error rates
            if os.path.isdir(os.path.join(training_dir, error_rate)):
                run_eval_loop(training_dir, error_rate)