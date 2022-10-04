#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=A6000
#SBATCH  --exclude='sumac'
#SBATCH  --gres=gpu:1
#SBATCH  --mem=48G
#SBATCH  --time=48:00:00
#SBATCH  --priority='TOP'

# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate placenta-segmentation-gpu-2

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/qcseg/automatic-placenta-segmentation/train_placenta.py \
--batch_size 8 \
--randomize_image_dataloader \
--run_number 0 \
--data_path '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/PIPPI_Data/split-nifti-processed/' \
--save_path '/data/scratch/nkarani/projects/qcseg/models/existing_folds_old_run0/' \
--which_data_split 'existing_folds'

echo "Hostname was: `hostname`"
echo "Reached end of job file."