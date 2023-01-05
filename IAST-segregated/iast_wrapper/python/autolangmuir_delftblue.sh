#!/bin/bash
#SBATCH --job-name="autolangmuir"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB

#SBATCH -N 1
#SBATCH --export=ALL
echo $SLURM_JOBID > jobid
valhost=$SLURM_JOB_NODELIST
echo $valhost > hostname
module load slurm
module load 2022r2
module load py-numpy 
module load py-scipy
module load py-matplotlib
echo "Starting autolangmuir with sbatch"
python autolangmuir.py 
echo "autolangmuir complete"