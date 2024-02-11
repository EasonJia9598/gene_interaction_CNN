python ./permutated_CNN/scripts/generate_data.py \
    --directory /home/eeepc/Documents/random_tree/random_tree_5_genes_2024_02_10_13_38_30_n_records_200000 \
    --profile_file_name_pattern profiles_data_5_genes_2500_records.csv \
    --rates_file_name_pattern rates_data_5_genes_2500_records.csv \
    --number_of_profiles_in_one_gene_image_array 32 \
    --generation_type 0 \
    --cut_off_files 0 \
    --total_files_to_convert 32 \
    --gene_image_type 0



#     parser.add_argument('--directory', type=str, help='directory for the raw data')
#     parser.add_argument('--profile_file_name_pattern', type=str, help='pattern for the profile file names')
#     parser.add_argument('--rates_file_name_pattern', type=str, help='pattern for the rates file names')
#     parser.add_argument('--number_of_profiles_in_one_gene_image_array', type=int, default=6, help='number of profiles in one gene image array')
#     parser.add_argument('--generation_type', type=int, default=0, help='generation type, 0 for generating genes images and rates, 1 for generating rates, 2 for generating gene images')
#     parser.add_argument('--cut_off_files', type=int, default=0, help='cut off files')
#     parser.add_argument('--total_files_to_convert', type=int, default=32, help='total files to convert')
#     parser.add_argument('--gene_image_type', type=int, default=0, help='gene image type, 0 for permutated gene images, 1 for duplicates gene images')

# # --directory /home/eeepc/Documents/random_tree/code_base/example_data/ \