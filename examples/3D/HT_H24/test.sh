#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/sliding_windowed_tiling/microsplit/MicroSplit-reproducibility/examples/logs_test_script/%x-%j.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1024GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=MST_H24_SW
#SBATCH --time=36:00:00

source ~/.bashrc
cd /home/aman.kukde/sliding_windowed_tiling/microsplit/
conda activate msr
python3.10 /home/aman.kukde/sliding_windowed_tiling/microsplit/MicroSplit-reproducibility/examples/3D/HT_H24/test_script_3d.py
