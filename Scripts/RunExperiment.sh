#!/bin/bash

echo "Starting job at $(date)"

# Job Identifier for logging
export RUN_ID=$1

# Environment variables setup
export HOME=$(pwd)
export ENVNAME=vmamba
export ENVDIR=$ENVNAME
export PATH=$ENVDIR/bin:$PATH
export DATASET_PATH=MICCAI_BraTS_2019_Data_Training
export OUTPUT_PATH=results

# Create the output directory
mkdir $OUTPUT_PATH

# Set up the environment
(
    cp /staging/vichare2/$ENVNAME.tar.gz ./
    mkdir $ENVDIR
    tar -xzf $ENVNAME.tar.gz -C $ENVDIR
    . $ENVDIR/bin/activate
    echo "Conda environment activated"
) &

# Copy the dataset
(
    cp /staging/vichare2/brats-2019.zip ./
    unzip -q brats-2019.zip
    echo "Dataset copied"
) &

# Wait for the environment and dataset to be copied
wait


# Validate the machine environment
echo "Home: $HOME"
echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
date
nvidia-smi
# cat /etc/os-release
# printenv

# Run the job
echo "Script execution started at $(date)"
python exp_with_dataset.py
# python run.py
# python optim.py
echo "Script execution ended at $(date)"


# Past command references
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# bash ~/miniconda.sh -b -p $HOME/miniconda
# echo $SHELL
# eval "$(/$HOME/miniconda/bin/conda shell.bash hook)"
# conda init
# conda env list

# conda env create -f environment.yml
# conda env list
# conda pack -n vmamba --dest-prefix='$ENVDIR'
# chmod 644 vmamba.tar.gz
# ls -sh vmamba.tar.gz

# echo conda: $(which conda)
# echo python: $(which python)
# echo pip: $(which pip)
# echo nvcc: $(which nvcc)
# nvcc --version
# echo gcc: $(which gcc)
# gcc --version
# echo g++: $(which g++)
# g++ --version


