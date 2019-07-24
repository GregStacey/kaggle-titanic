#!/bin/bash
#SBATCH --time=00:55:00
#SBATCH -e titanic_slurm.err
#SBATCH --mail-user=richard.greg.stacey@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o titanic_slurm.out
#SBATCH --mem=2G
#SBATCH --account=def-ljfoster
#SBATCH --array=1-600

### Random grid search of all hyperparameters

module load gcc/7.3.0
module load r/3.4.4

PROJECT_DIR=~/projects/def-ljfoster/rstacey/kaggle-titanic/
cd ${PROJECT_DIR}

Rscript ${PROJECT_DIR}/R/titanic_slurm.R $SLURM_ARRAY_TASK_ID
