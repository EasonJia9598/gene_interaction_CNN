#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:55:00
#SBATCH --job-name data_generation_job
#SBATCH --output=/****/zeshengj/logs/5_genes_000_tips_100_image_data_generation_output_%j.txt
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=ALL

module load NiaEnv/2019b python/3.11.5
source ~/.virtualenvs/myenv/bin/activate

# cd /****/zeshengj/CNN/gene_corelation_CNN

# sh /****/zeshengj/CNN/gene_corelation_CNN/data_generation.sh

# sh /****/zeshengj/CNN/data/simulations_data/400_tips_10_genes_400_image_size/data_generation.sh

# sh /****/zeshengj/CNN/data/simulations_data/2000_tips_tree_seed_26675_5_genes_1600000_records_2024_04_01_04_45_23/5_genes_2000_tips_100_image_data_generation.sh

sh /****/zeshengj/CNN/data/simulations_data/3000_tips_tree_seed_26675_10_genes_1600000_records_2024_04_07_20_36_06/10_genes_3000_tips_data_generation.sh