import os
import sys
import json
import uuid
import shutil
import subprocess

# Script to generate training folder structure from json training config file -------------------
# TODO : script doesn't copy static decoder
# TODO : script doesn't perform anychecks if structure is correctly generated

if len(sys.argv) != 2:
  print("Usage: python3 make_experiment.py <training_config.json>")
  sys.exit(1)

# identifier for training to differentiate between runs at same error rate
training_id = str(uuid.uuid4())[0:6]

scripts_dir = os.path.join(os.getcwd(), "experiments", "scripts")

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

# add training ID to configuration
config["training_id"] = training_id


# 1. Create base directory in /cluster_scripts/experiments
rootdir = f"experiments/{config['config_dir']}"
os.mkdir(rootdir)

# 2. Copy (updated with ID) config file to root
with open(f"{rootdir}/training_config.json", "w") as f:
  json.dump(config, f)

# 3. Create folder structure and mandatory files
os.mkdir(f"{rootdir}/results")
open(f"{rootdir}/history.txt", "w").close()

current_error_rate = config['base_param_grid']['p_phys']
with open(f"{rootdir}/current_error_rate.txt", "w") as f:
  f.write(f"{current_error_rate}\n")

shutil.copyfile(f"{scripts_dir}/Controller.py", f"{rootdir}/Controller.py")

for phy in config["controller_params"]["p_phys_list"]:
  os.mkdir(f"{rootdir}/{phy}")
  os.mkdir(f"{rootdir}/{phy}/output_files")

  if phy == current_error_rate:
    shutil.copyfile(f"{scripts_dir}/Generate_Base_Configs_and_Simulation_Scripts.py", f"{rootdir}/Generate_Base_Configs_and_Simulation_Scripts.py")
    shutil.copyfile(f"{scripts_dir}/Start_Simulations.sh", f"{rootdir}/Start_Simulations.sh")
    shutil.copyfile(f"{scripts_dir}/Single_Point_Training_Script.py", f"{rootdir}/{phy}/Single_Point_Training_Script.py")
  else:
    shutil.copyfile(f"{scripts_dir}/Single_Point_Continue_Training_Script.py", f"{rootdir}/{phy}/Single_Point_Continue_Training_Script.py")
  shutil.copyfile(f"{scripts_dir}/Single_Point_Testing_Script.py", f"{rootdir}/{phy}/Single_Point_Testing_Script.py")

# 4. Move to new working environment
os.chdir(rootdir)
# Generate configs for first error rate
subprocess.run(["python3", "Generate_Base_Configs_and_Simulation_Scripts.py"])
# Make Start_Simulations.sh executable (0o is octinteger literal)
os.chmod("Start_Simulations.sh", 0o744)