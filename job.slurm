#!/bin/bash
#SBATCH --job-name=testing!        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=250G         # memory per cpu-core (4G is default)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=fc1991@princeton.edu

module purge
module load anaconda3/2022.10
conda activate test

python test.py
