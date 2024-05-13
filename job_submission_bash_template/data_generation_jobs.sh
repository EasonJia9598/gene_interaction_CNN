#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:55:00
#SBATCH --job-name data_generation_job
#SBATCH --output=/****/zeshengj/logs/5_genes_000_tips_100_image_data_generation_output_%j.txt
#SBATCH --mail-user=z****@dal.ca
#SBATCH --mail-type=ALL

module load NiaEnv/2019b python/3.11.5
source ~/.virtualenvs/myenv/bin/activate


python ./permutated_CNN/scripts/generate_data.py \
    --directory ./data/simulations_data/100_Darwin_scenerios_normal_tips_5_genes_tree_seed_23123_8000_records_2024_05_10_12_14_51 \
    --profile_file_name_pattern profiles.csv \
    --rates_file_name_pattern rates.csv \
    --number_of_profiles_in_one_gene_image_array 80 \
    --generation_type 0 \
    --cut_off_files 0 \
    --total_files_to_convert 40 \
    --gene_image_type 1\
    --number_of_genes 5 \
    --image_width 100