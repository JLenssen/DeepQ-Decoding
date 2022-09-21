# ------------ This script generates the initial grid over which we will start our search ----------------

import os
import json
import shutil
import pickle
cwd = os.getcwd()

# ------------ the fixed parameters: These are constant for all error rates -----------------------------

with open('training_config.json', 'r') as f:
    data = json.load(f)
    fixed_config = data["fixed_config"]
    parameter_grid = data["base_param_grid"]

fixed_config_path = os.path.join(cwd, "fixed_config.p")
pickle.dump(fixed_config, open(fixed_config_path, "wb" ) )

# Get initial error rate for which we generate initial configurations
initial_error_rate = str(parameter_grid["p_phys"])
configs_path = os.path.join(cwd, initial_error_rate)

# ---------- Generate training scripts based on parameter grid ----------------------------------------------------

config_counter = 1
for ls in parameter_grid["learning_starts_list"]:
    for lr in parameter_grid["learning_rate_list"]:
        for ef_count, ef in enumerate(parameter_grid["exploration_fraction_list"]):
            for me in parameter_grid["max_eps_list"]:
                for tnuf in parameter_grid["target_network_update_freq_list"]:
                    for g in parameter_grid["gamma_list"]:
                        for fe in parameter_grid["final_eps_list"]:

                            variable_config_dict = {"p_phys": parameter_grid["p_phys"],
                            "p_meas": parameter_grid["p_phys"],
                            "success_threshold": parameter_grid["success_threshold"],
                            "learning_starts": ls,
                            "learning_rate": lr,
                            "exploration_fraction": ef,
                            "max_eps": me,
                            "target_network_update_freq": tnuf,
                            "gamma": g,
                            "final_eps": fe}

                            config_directory = os.path.join(configs_path, "config_"+str(config_counter)+"/")
                            if not os.path.exists(config_directory):
                                os.makedirs(config_directory)
                            else:
                                shutil.rmtree(config_directory)           #removes all the subdirectories!
                                os.makedirs(config_directory)

                            file_path = os.path.join(config_directory, "variable_config_"+str(config_counter) + ".p")
                            pickle.dump(variable_config_dict, open(file_path, "wb" ) )
                                        
                            # Now, write into the bash script exactly what we want to appear there
                            job_limit = str(parameter_grid["sim_time_per_ef"][ef_count])
                            job_name = data["training_id"]+"_"+str(parameter_grid["p_phys"])+"_"+str(config_counter)
                            job_cores = fixed_config["job_cores"]
                            job_memory_per_cpu = fixed_config["job_memory_per_cpu"]
                            output_file = os.path.join(configs_path,"output_files/out_"+job_name+".out")
                            error_file = os.path.join(configs_path,"output_files/err_"+job_name+".err")
                            training_script = os.path.join(configs_path, "Single_Point_Training_Script.py")
                            testing_script = os.path.join(configs_path, "Single_Point_Testing_Script.py")


                            f = open(config_directory + "/simulation_script.sh",'w')  
                            f.write('''#!/bin/bash
 
#SBATCH --job-name={job_name}              # Job name, will show up in squeue output
#SBATCH --ntasks={job_cores}               # Number of cores
#SBATCH --nodes=1                          # Ensure that all cores are on one machine
#SBATCH --time=0-{job_limit}:30:00         # Runtime in DAYS-HH:MM:SS format # TODO fix me
#SBATCH --mem-per-cpu={job_memory_per_cpu} # Memory per cpu in MB (see also --mem) 
#SBATCH --output={output_file}             # File to which standard out will be written
#SBATCH --error={error_file}               # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

module load miniconda
source activate deepq-mkl

# ----------- Tensorflow XLA flag -----------------------------------------------
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# ------- run the script if argument given, we only test the agent --------------
if [ $# -eq 0 ]; then
    python -u {training_script} {config_counter} {configs_path} || exit 1
fi
python -u {testing_script} {config_counter} {configs_path} || exit 1

#----------- wait some time ------------------------------------

sleep 50'''.format(job_name=job_name,
                output_file=output_file,
                error_file=error_file,
                job_limit=job_limit,
                job_cores=job_cores,
                job_memory_per_cpu=job_memory_per_cpu,
                training_script=training_script,
                testing_script=testing_script,
                config_counter=config_counter,
                configs_path=configs_path))
                            f.close()
                            config_counter += 1 
