#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:45
#SBATCH --job-name training_job
#SBATCH --output=/***/zeshengj/logs/prediction_output_%j.txt
#SBATCH --mail-user=z****@dal.ca
#SBATCH --mail-type=ALL

module load anaconda3
source activate genes_env

python ./permutated_CNN/scripts/predict.py \
    --main_dir ./data/simulations_data/100_Darwin_scenerios_normal_tips_5_genes_tree_seed_23123_8000_records_2024_05_10_12_14_51/\
    --model_checkpoint_path ./data/simulations_data/100_normal_tips_5_genes_tree_seed_23123_2000000_records_2024_05_09_22_10_39/model_checkpoints/05_10_02_40_13_5_genes_100_normal_tips_ResBet34_img_batch_156_sub_batch_1_1e-4/batch_best_model_0.13.pth \
    --batch_size 20 \
    --num_outputs 20 \
    --gene_image_path ./data/simulations_data/100_Darwin_scenerios_normal_tips_5_genes_tree_seed_23123_8000_records_2024_05_10_12_14_51/gene_images/duplicates_gene_images/0_random_tree_gene_image.npy\
    --gene_rates_path ./data/simulations_data/100_Darwin_scenerios_normal_tips_5_genes_tree_seed_23123_8000_records_2024_05_10_12_14_51/rates/0_concatenated_rates.csv\
    --ResNet_depth 34 \
    --image_length 100 \
    --padding_length 100 \

# parser = argparse.ArgumentParser(description="Train a CNN model for regression")
    # parser.add_argument("--main_dir", type=str, help="Directory containing the required files")
    # parser.add_argument("--model_checkpoint_path", type=str, help="Model checkpoint path")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    # parser.add_argument("--num_outputs", type=int, default=10, help="Number of outputs for the regression model")
    # parser.add_argument("--gene_image_path", type=str, help="gene image path")
    # parser.add_argument("--gene_rates_path", type=str, help="gene rates path")
    # parser.add_argument("--ResNet_depth", type=int, default=50, help="ResNet depth")