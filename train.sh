python /****/zeshengj/CNN/gene_corelation_CNN/permutated_CNN/scripts/train.py \
    --main_dir /****/zeshengj/CNN/data/100_tips_tree_40_genes_8000000_records_2024_03_12_10_39_00  \
    --if_use_temperary_ssd 0\
    --temperary_ssd_dr /****/zeshengj/CNN/data/100_tips_tree_40_genes_1200000_records_2024_03_08_08_16_15/temperary_folder \
    --load_model_checkpoint 1 \
    --model_checkpoint_path /****/zeshengj/CNN/data/100_tips_tree_40_genes_8000000_records_2024_03_12_10_39_00/model_checkpoints/03_14_21_11_21_100_genes_ResNet50_2e-5/batch_best_model_0.677.pth \
    --epochs 100000 \
    --batch_size 512 \
    --num_outputs 780 \
    --sub_training_batch 2\
    --input_gene_image_size "1, 400, 200" \
    --gene_image_type 1 \
    --log_file_name "2nd_40_genes_ResBet50_batch_512_sub_batch_1_1e-4" \
    --model_type "ResNet" \
    --ResNet_depth 50 \
    --learning_rate 1e-4 \

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
    # parser.add_argument("--sub_training_batch", type=int, default=10, help="Number of sub-training epochs for 1 batch")
    # parser.add_argument('--input_gene_image_size', metavar='N', type=str, nargs=3, help='Three integers for the input gene image size. Ex: \'(1, 100, 600)\'')
    # parser.add_argument("--gene_image_type", type=int, default=0, help="gene image type, 0 for permutated gene images, 1 for duplicates gene images")
    #  parser.add_argument("--model_type", type=str, default="CNN", help="Model type, CNN or ResNet")