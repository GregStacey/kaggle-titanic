#!/bin/bash
#SBATCH --time=23:55:00
#SBATCH -e titanic.err
#SBATCH --mail-user=richard.greg.stacey@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o titanic.out
#SBATCH --mem=2G
#SBATCH --array=1-1000
#SBATCH --account=def-ljfoster

### Random grid search of all hyperparameters

module load gcc/7.3.0
module load rdkit/2018.03.3

PROJECT_DIR=~/projects/def-ljfoster/rstacey/kaggle-titanic/
cd ${PROJECT_DIR}

Rscript ${PROJECT_DIR}/R/titanic.R
