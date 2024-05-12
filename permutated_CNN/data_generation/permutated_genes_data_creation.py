from permutated_CNN.data_generation.gene_to_image_converter import *
from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial
import multiprocessing as mp
import time
import pandas as pd

#####################################################
# extract_numeric_part
#####################################################
def extract_numeric_part(filename):
    """
    Extracts the numeric part from a given filename.
    
    Parameters:
        filename (str): The name of the file from which to extract the numeric part.
        
    Returns:
        int: The numeric part extracted from the filename.
    """
    return int(''.join(filter(str.isdigit, filename)))


#####################################################
# read_csv_with_chunksize
#####################################################
def read_csv_with_chunksize(file_path, chunksize=100):
    """
    Reads a CSV file in chunks and returns a single DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.
    chunksize (int, optional): The number of rows to read per chunk. Defaults to 100.

    Returns:
    pandas.DataFrame: A DataFrame containing the concatenated data from all chunks.

    """
    # Get the total number of lines in the file
    num_lines = sum(1 for _ in open(file_path))

    # Create an empty list to store the chunks
    chunks = []

    # Create a progress bar
    with tqdm(total=num_lines) as pbar:
        # Read the CSV file in chunks and iterate over each chunk
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            # Append the processed chunk to the list
            chunks.append(chunk)
            pbar.update(len(chunk))

    # Concatenate the list of chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)
    return df


#####################################################
# concatenate_gene_rates
#####################################################
def concatenate_gene_rates(files, raw_data_path, _epochs, saving_folder):
    """
    Concatenates multiple gene rate files into a single dataframe and saves it to a CSV file.

    Parameters:
    - files (list): A list of file names to be concatenated.
    - raw_data_path (str): The path to the directory containing the gene rate files.
    - _epochs (int): The number of epochs.
    - saving_folder (str): The path to the directory where the concatenated data will be saved.

    Returns:
    None (CSV file is saved to disk)
    """
    
    index = 1
    dataframes = []
    for file in files:
        file_path = os.path.join(raw_data_path, file)
        df = pd.read_csv(file_path , low_memory=False)
        print(file_path)
        dataframes.append(df)
        print(f"finish file {index}")
        index += 1
    # Save the concatenated data to a new CSV file
    concatenated_data = pd.concat(dataframes,  axis=0)
    concatenated_data.to_csv(f'{saving_folder}/{_epochs}_concatenated_rates.csv', index=False)


#####################################################
# concatenate_permutated_gene_profiles
#####################################################
def concatenate_permutated_gene_profiles(files, raw_data_path, _epochs, permuations_result, saving_folder):
    """
    Concatenates permutated gene profiles from multiple files into a single numpy array.

    Parameters:
    - files (list): A list of file names containing permutated gene profiles.
    - raw_data_path (str): The path to the directory where the files are located.
    - _epochs (int): The number of epochs.
    - permuations_result (object): The result of the permutations.
    - saving_folder (str): The path to the directory where the concatenated gene profiles will be saved.

    Returns:
    - None (CSV file is saved to disk)
    """
    progress_bar = tqdm(total=len(files), desc='Generating Files')
    file_index = 0 
    for file in files:
        file_path = os.path.join(raw_data_path, file)
        print(file_path)
        X_profiles = pd.read_csv(file_path)
        gene_image_df = create_gene_image_dataset_with_permutation(X_profiles, permuations_result)
        if file_index == 0:
            previous_gene_image = gene_image_df
        else:
            print("############################################################################")
            print(" CONCATENATING FILES")
            previous_gene_image = np.concatenate([previous_gene_image, gene_image_df],  axis = 0 )
            print("############################################################################")

        progress_bar.update(1)
        file_index += 1
    
    progress_bar.close()
    np.save(f'{saving_folder}/{_epochs}_random_tree_permutated_gene_image.npy', previous_gene_image)


#####################################################
# concatenate_duplicates_gene_profiles
#####################################################
def concatenate_duplicates_gene_profiles(image_width, number_of_genes, files, raw_data_path, _epochs, saving_folder):
    """
    Concatenates gene profiles from multiple files into a single dataset.

    Args:
        image_width (int): The width of the gene image.
        number_of_genes (int): The number of genes in each image.
        files (list): A list of file names containing gene profiles.
        raw_data_path (str): The path to the directory containing the gene profile files.
        _epochs (int): The number of epochs.
        saving_folder (str): The path to the directory where the concatenated dataset will be saved.

    Returns:
        None
    """
    progress_bar = tqdm(total=len(files), desc='Generating Files')
    file_index = 0 
    for file in files:
        file_path = os.path.join(raw_data_path, file)
        # print(file_path)
        X_profiles = pd.read_csv(file_path, low_memory=False)
        print("############################################################################")
        print("File path: ", file_path, " \nFile Shape: ", X_profiles.shape)
        print("############################################################################")
        gene_image_df = create_gene_image_dataset_duplicates(number_of_genes, X_profiles, int(image_width / number_of_genes))
        print("############################################################################")
        print("File Shape after duplicates: ", gene_image_df.shape)
        print("############################################################################")
        if file_index == 0:
            previous_gene_image = gene_image_df
        else:
            print("############################################################################")
            print(" CONCATENATING FILES")
            print("File Shape before: ", previous_gene_image.shape)
            print("############################################################################")
            previous_gene_image = np.concatenate([previous_gene_image, gene_image_df],  axis = 0 )
            print("############################################################################")
            print("File Shape after: ", previous_gene_image.shape)
            print("############################################################################")

        progress_bar.update(1)
        file_index += 1
    
    progress_bar.close()
    np.save(f'{saving_folder}/{_epochs}_random_tree_gene_image.npy', previous_gene_image)


#####################################################
# combine_rates
#####################################################
def combine_rates(directory, file_name_pattern, number_of_rates_in_one_gene_image_array=6, cut_off_files=False, total_files_to_convert=32):
    """
    Combines rates from multiple files into gene image arrays.

    Args:
        directory (str): The directory path where the raw data and rates will be saved.
        file_name_pattern (str): The pattern to match the files in the raw data directory.
        number_of_rates_in_one_gene_image_array (int, optional): The number of rates to combine into one gene image array. Defaults to 6.
        cut_off_files (bool, optional): Whether to cut off files after a certain number. Defaults to False.
        total_files_to_convert (int, optional): The total number of files to convert if cut_off_files is True. Defaults to 32.

    Returns:
        None
    """
    # Get a list of all files in the directory
    raw_data_path = f'{directory}/raw_data/rates'
    saving_folder = f'{directory}/rates'
    
    try:
        print("Create rates directory")
        os.mkdir(saving_folder)
    except:
        print("Directory already exists")

    files = os.listdir(raw_data_path)

    # Filter files to include only those that match the pattern
    files = [file for file in files if file.endswith(file_name_pattern)]

    # Sort the files based on their numeric values
    files.sort(key=extract_numeric_part)
    
    if cut_off_files is True:
        print("############################################################################")
        print("CUTTING OFF FILES")
        files = files[:total_files_to_convert]

    num_batches = number_of_rates_in_one_gene_image_array
    number_of_gene_arrays = int(len(files) / num_batches)

    for _epochs in tqdm(range(number_of_gene_arrays)):
        sub_files = files[_epochs * num_batches: (_epochs + 1) * num_batches]
        concatenate_gene_rates(sub_files, raw_data_path, _epochs, saving_folder)

    # In case there are more files than the number of gene arrays
    if len(files) > number_of_gene_arrays * num_batches:
        sub_files = files[number_of_gene_arrays * num_batches:]
        concatenate_gene_rates(sub_files, raw_data_path, number_of_gene_arrays, saving_folder)


#####################################################
# create_permutated_gene_images
#####################################################
def create_permutated_gene_images(directory, profile_file_name_pattern, number_of_profiles_in_one_gene_image_array=6, cut_off_files=False, total_files_to_convert=32):
    """
    Create permutated gene images from raw data profiles.

    Args:
        directory (str): The directory where the files are located.
        profile_file_name_pattern (str): The pattern to match the profile files.
        number_of_profiles_in_one_gene_image_array (int, optional): The number of profiles to include in one gene image array. Defaults to 6.
        cut_off_files (bool, optional): Whether to cut off files after a certain number. Defaults to False.
        total_files_to_convert (int, optional): The total number of files to convert. Defaults to 32.

    Returns:
        None
    """
    # Specify the directory where your files are located
    raw_data_path = f'{directory}/raw_data/profiles'
    saving_folder = f'{directory}/gene_images/permutated_gene_images'

    try:
        print("Create images directory")
        os.mkdir(saving_folder)
    except:
        print("Directory already exists")


    # Get a list of all files in the directory
    files = os.listdir(raw_data_path)

    # Filter files to include only those that match the pattern
    files = [file for file in files if file.endswith(profile_file_name_pattern)]

    # Sort the files based on their numeric values
    files.sort(key=extract_numeric_part)

    if cut_off_files is True:
        files = files[:total_files_to_convert]

    # get permutated CNN sequence
    permuations_result =  generate_permutations()
    shuffled_permutations = np.random.permutation(permuations_result)
    permuations_result = (shuffled_permutations.flatten()[:200]).reshape(80, 5)

    num_batches = number_of_profiles_in_one_gene_image_array

    number_of_gene_arrays = int(len(files) / num_batches)
    

    for _epochs in tqdm(range(number_of_gene_arrays)):
        sub_files = files[_epochs * num_batches: (_epochs + 1) * num_batches]
        concatenate_permutated_gene_profiles(sub_files, raw_data_path, _epochs, permuations_result, saving_folder)

    # In case there are more files than the number of gene arrays
    if len(files) > number_of_gene_arrays * num_batches:
        sub_files = files[number_of_gene_arrays * num_batches:]
        concatenate_permutated_gene_profiles(sub_files, raw_data_path, number_of_gene_arrays, permuations_result, saving_folder)


#####################################################
# create_duplicates_gene_images
#####################################################
def create_duplicates_gene_images(image_width, number_of_genes, directory, profile_file_name_pattern, number_of_profiles_in_one_gene_image_array=6, cut_off_files=False, total_files_to_convert=32):
    """
    Creates duplicate gene images by concatenating multiple gene profiles.

    Parameters:
    - image_width (int): The width of the gene images.
    - number_of_genes (int): The number of genes to include in each gene image.
    - directory (str): The directory where the raw data and gene images will be saved.
    - profile_file_name_pattern (str): The pattern to match the profile file names.
    - number_of_profiles_in_one_gene_image_array (int): The number of profiles to include in each gene image array. Default is 6.
    - cut_off_files (bool): Whether to limit the number of files to convert. Default is False.
    - total_files_to_convert (int): The total number of files to convert if cut_off_files is True. Default is 32.

    Returns:
    None
    """
    # Specify the directory where your files are located
    raw_data_path = f'{directory}/raw_data/profiles'
    saving_folder = f'{directory}/gene_images/duplicates_gene_images'

    try:
        print("Create images directory")
        os.mkdir(saving_folder)
    except:
        print("Directory already exists")

    # Get a list of all files in the directory
    files = os.listdir(raw_data_path)

    # Filter files to include only those that match the pattern
    files = [file for file in files if file.endswith(profile_file_name_pattern)]

    # Sort the files based on their numeric values
    files.sort(key=extract_numeric_part)

    if cut_off_files is True:
        files = files[:total_files_to_convert]

    num_batches = number_of_profiles_in_one_gene_image_array
    number_of_gene_arrays = int(len(files) / num_batches)
    

    for _epochs in tqdm(range(number_of_gene_arrays)):
        sub_files = files[_epochs * num_batches: (_epochs + 1) * num_batches]
        concatenate_duplicates_gene_profiles(image_width, number_of_genes, sub_files, raw_data_path, _epochs, saving_folder)
        
    # In case there are more files than the number of gene arrays
    if len(files) > number_of_gene_arrays * num_batches:
        sub_files = files[number_of_gene_arrays * num_batches:]
        concatenate_duplicates_gene_profiles(image_width, number_of_genes, sub_files, raw_data_path, number_of_gene_arrays, saving_folder)


#####################################################
# data_generation
#####################################################
def data_generation(image_width, number_of_genes, directory, profile_file_name_pattern, rates_file_name_pattern, number_of_files_in_one_gene_image_array=6, generation_type=0, cut_off_files=False, total_files_to_convert=32, gene_image_type=0):
    """
    Generates data for gene images.

    Parameters:
    - image_width (int): The width of the gene images.
    - number_of_genes (int): The number of genes.
    - directory (str): The directory where the data will be generated.
    - profile_file_name_pattern (str): The pattern for the profile file names.
    - rates_file_name_pattern (str): The pattern for the rates file names.
    - number_of_files_in_one_gene_image_array (int): The number of files in one gene image array. Default is 6.
    - generation_type (int): The type of data generation. Default is 0.
    - cut_off_files (bool): Whether to cut off files. Default is False.
    - total_files_to_convert (int): The total number of files to convert. Default is 32.
    - gene_image_type (int): The type of gene image. Default is 0.

    Returns:
    None
    """
    try:
        os.mkdir(f'{directory}/gene_images')
        print("gene_images Directory created successfully")
    except:
        print("Directory already exists")

    if generation_type == 0 or generation_type == 1:
        print("############################################################################")
        print("COMBINING RATES")
        print("############################################################################")
        # Convert the gene rates into one csv
        combine_rates(directory, rates_file_name_pattern, number_of_files_in_one_gene_image_array, cut_off_files, total_files_to_convert)

    if generation_type == 0 or generation_type == 2:
        print("############################################################################")
        print("CREATING GENE IMAGES")
        print("############################################################################")
        
        # create gene images
        if gene_image_type == 0:
            print("############################################################################")
            print("PERMUTATED GENE IMAGES")
            print("############################################################################")
            create_permutated_gene_images(directory, profile_file_name_pattern, number_of_files_in_one_gene_image_array, cut_off_files, total_files_to_convert)
        else:
            print("############################################################################")
            print("DUPLICATES GENE IMAGES")
            print("############################################################################")
            create_duplicates_gene_images(image_width, number_of_genes, directory, profile_file_name_pattern, number_of_files_in_one_gene_image_array, cut_off_files, total_files_to_convert)

    print("############################################################################")
    print("Data Generation Completed")
    print("############################################################################")
