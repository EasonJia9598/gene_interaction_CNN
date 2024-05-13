#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:45
#SBATCH --job-name training_job
#SBATCH --output=/****/zeshengj/logs/5_genes_1000_padding_tips_2nd_trainin_sub_1_output_%j.txt
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=ALL

module load anaconda3
source activate genes_env

python /****/zeshengj/CNN/gene_corelation_CNN/permutated_CNN/scripts/train.py \
    --main_dir /****/zeshengj/CNN/data/simulations_data/1000_padding_tips_5_genes_tree_seed_23123_1600000_records_2024_05_03_09_25_36 \
    --if_use_temperary_ssd 0\
    --temperary_ssd_dr none \
    --load_model_checkpoint 1 \
    --model_checkpoint_path /****/zeshengj/CNN/data/simulations_data/1000_padding_tips_5_genes_tree_seed_23123_1600000_records_2024_05_03_09_25_36/model_checkpoints/05_04_00_28_19_5_genes_1000_padding_tips_1st_ResBet34_img_batch_156_sub_batch_1_1e-4/batch_best_model_0.062.pth \
    --epochs 100000 \
    --batch_size 128 \
    --num_outputs 10 \
    --sub_training_batch 1\
    --input_gene_image_size "1, 1000, 200" \
    --gene_image_type 1 \
    --log_file_name "5_genes_1000_padding_tips_2nd_ResBet34_img_batch_156_sub_batch_1_1e-4" \
    --model_type "ResNet" \
    --ResNet_depth 34 \
    --learning_rate 1e-4\