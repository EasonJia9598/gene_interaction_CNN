python ./permutated_CNN/scripts/train.py \
    --main_dir /media/eeepc/3866FF5566FF127C/uniform_datta/uniform_random_tree_5_genes_4000000_records_2024_02_15_01_30_28  \
    --temperary_ssd_dr /home/eeepc/Documents/random_tree/code_base/temperary_data_saver/pure_uniform_4million \
    --epochs 100000 \
    --batch_size 64 \
    --num_outputs 10 \
    --sub_training_batch 6 \
    --input_gene_image_size "1, 100, 200" \
    --gene_image_type 1 \
    --log_file_name "random_sampled_pure_uniform_permutated_ResNet_34_1e3" \
    --model_type "ResNet" \
    --ResNet_depth 50 \
    --learning_rate 2e-4 \


# gene_image_type 1 for duplicates images, 0 for permutated images



# python ./permutated_CNN/scripts/train.py \
#     --main_dir /home/eeepc/Documents/random_tree/random_tree_5_genes_2024_02_10_13_38_30_n_records_200000 \
#     --epochs 100000 \
#     --batch_size 64 \
#     --num_outputs 10 \
#     --sub_training_batch 5 \
#     --input_gene_image_size "1, 100, 600" \
#     --gene_image_type 0 \
#     --log_file_name "Permutated_CNN_200K"



# --directory /home/eeepc/Documents/random_tree/code_base/example_data/ \
    # parser.add_argument("--main_dir", type=str, help="Directory containing the required files")
    # parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    # parser.add_argument("--num_outputs", type=int, default=10, help="Number of outputs for the regression model")
    # parser.add_argument("--sub_training_batch", type=int, default=10, help="Number of sub-training batches")
    # parser.add_argument('--input_gene_image_size', metavar='N', type=str, nargs=3, help='Three integers for the input gene image size. Ex: \'(1, 100, 600)\'')
    # parser.add_argument("--gene_image_type", type=int, default=0, help="gene image type, 0 for permutated gene images, 1 for duplicates gene images")
    #  parser.add_argument("--model_type", type=str, default="CNN", help="Model type, CNN or ResNet")