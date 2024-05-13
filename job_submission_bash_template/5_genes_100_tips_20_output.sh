#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:45
#SBATCH --job-name training_job
#SBATCH --output=/***88/zeshengj/logs/5_genes_100_normal_tips_ResBet34_img_batch_156_sub_batch_1_1e-4_output_%j.txt
#SBATCH --mail-user=z****@dal.ca
#SBATCH --mail-type=ALL

module load anaconda3
source activate genes_env

python /*****/zeshengj/CNN/gene_corelation_CNN/permutated_CNN/scripts/train.py \
    --main_dir /*****/zeshengj/CNN/data/simulations_data/100_normal_tips_5_genes_tree_seed_23123_2000000_records_2024_05_09_22_10_39 \
    --if_use_temperary_ssd 0\
    --temperary_ssd_dr none \
    --load_model_checkpoint 0 \
    --model_checkpoint_path none \
    --epochs 100000 \
    --batch_size 1024 \
    --num_outputs 10 \
    --sub_training_batch 2\
    --input_gene_image_size "1, 100, 100" \
    --gene_image_type 1 \
    --log_file_name "5_genes_100_normal_tips_ResBet34_img_batch_156_sub_batch_1_1e-4" \
    --model_type "ResNet" \
    --ResNet_depth 34 \
    --learning_rate 1e-4\



