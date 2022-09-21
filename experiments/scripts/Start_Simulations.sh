#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error rate directory not supplied, aborting."
    exit 1
fi

directory=$1

for d in ${directory}/* ; do
    script_path="${d}/simulation_script.sh"
    if [ -f $script_path ]; then
        sbatch $script_path
    fi
done