
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
    args = parser.parse_args()

    if args.directory is None or args.profile_file_name_pattern is None or args.rates_file_name_pattern is None:
        parser.print_help()
        exit(1)

    directory = args.directory
    profile_file_name_pattern = args.profile_file_name_pattern
    rates_file_name_pattern = args.rates_file_name_pattern
    number_of_profiles_in_one_gene_image_array = int(args.number_of_profiles_in_one_gene_image_array)
    generatin_type = int(args.generation_type)

    print("############################################################################")
    print("Data Generation Started")
    print(f"Directory: {directory} ")
    print(f"Profile File Name Pattern: {profile_file_name_pattern}")
    print(f"Rates File Name Pattern: {rates_file_name_pattern}")
    print(f"Number of Profiles in One Gene Image Array: {number_of_profiles_in_one_gene_image_array}")
    print("############################################################################")
    data_generation(directory, profile_file_name_pattern, rates_file_name_pattern, number_of_profiles_in_one_gene_image_array, generatin_type)



# Running 

if __name__ == "__main__":
    main()

