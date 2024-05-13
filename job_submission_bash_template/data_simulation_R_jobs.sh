#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=23:59:00
#SBATCH --job-name data_simulation_job
#SBATCH --output=/****/zeshengj/logs/5_genes_100_normal_tips_data_simulation_output_%j.txt
#SBATCH --mail-user=z****@dal.ca
#SBATCH --mail-type=ALL

module load gcc/8.3.0
module load intel/2019u4
module load r/4.1.2

source ~/.virtualenvs/R_env/bin/activate

Rscript ./R_code_simulate_data/data_simulation.R
