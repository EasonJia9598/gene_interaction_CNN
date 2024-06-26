
from permutated_CNN.model.CNN_structure import *
from permutated_CNN.data_generation.permutated_genes_data_creation import *
import argparse


def main():
    # import argparse
    parser = argparse.ArgumentParser(description='Data Generation')
    # add parser helper
    parser.add_argument('--directory', type=str, help='directory for the raw data')
    parser.add_argument('--profile_file_name_pattern', type=str, help='pattern for the profile file names')
    parser.add_argument('--rates_file_name_pattern', type=str, help='pattern for the rates file names')
    parser.add_argument('--number_of_profiles_in_one_gene_image_array', type=int, default=6, help='number of profiles in one gene image array')
    parser.add_argument('--generation_type', type=int, default=0, help='generation type, 0 for generating genes images and rates, 1 for generating rates, 2 for generating gene images')
    parser.add_argument('--cut_off_files', type=int, default=0, help='cut off files')
    parser.add_argument('--total_files_to_convert', type=int, default=32, help='total files to convert')
    parser.add_argument('--gene_image_type', type=int, default=0, help='gene image type, 0 for permutated gene images, 1 for duplicates gene images')
    parser.add_argument('--number_of_genes', type=int, default=6, help='number of genes')
    parser.add_argument('--image_width', type=int, default=6, help='image_width')


    args = parser.parse_args()

    if args.directory is None or args.profile_file_name_pattern is None or args.rates_file_name_pattern is None:
        parser.print_help()
        exit(1)

    directory = args.directory
    profile_file_name_pattern = args.profile_file_name_pattern
    rates_file_name_pattern = args.rates_file_name_pattern
    number_of_profiles_in_one_gene_image_array = int(args.number_of_profiles_in_one_gene_image_array)
    generatin_type = int(args.generation_type)
    gene_image_type = int(args.gene_image_type)

    
    if args.cut_off_files == 0:
        cut_off_files = False
    else:
        cut_off_files = True

    total_files_to_convert = int(args.total_files_to_convert)

    print("############################################################################")
    print("Data Generation Started")
    print(f"Directory is : {directory}")
    print(f"Profile File Name Pattern: {profile_file_name_pattern}")
    print(f"Rates File Name Pattern: {rates_file_name_pattern}")
    print(f"Number of Profiles in One Gene Image Array: {number_of_profiles_in_one_gene_image_array}")
    print(f"Generation Type: {generatin_type}")
    print(f"Cut Off Files: {cut_off_files}")
    if cut_off_files:
        print(f"Total Files to Convert: {total_files_to_convert}")
    print("############################################################################")
    data_generation(args.image_width, args.number_of_genes, directory, profile_file_name_pattern, rates_file_name_pattern, number_of_profiles_in_one_gene_image_array, generatin_type, cut_off_files, total_files_to_convert, gene_image_type)



# Running 

if __name__ == "__main__":
    main()

