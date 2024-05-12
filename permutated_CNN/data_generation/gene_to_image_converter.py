import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import pdb

#####################################################
# draw_gene
#####################################################
def draw_gene(gene):
    """
    Display an image representation of a gene.

    Parameters:
    gene (numpy.ndarray): The gene as a NumPy array.

    Returns:
    None
    """

    # Set the figure size
    fig, ax = plt.subplots(figsize=(80, 80))  # You can adjust the values (width, height) in inches

    # Display the image using Matplotlib
    ax.imshow(gene, cmap='gray')  # 'gray' colormap for grayscale images
    ax.set_title('Image from NumPy Array')
    # Show the plot
    plt.show()

#####################################################
# get_single_gene
#####################################################   
def get_single_gene(number_of_genes, profiles, index):
    """
    Extracts a single gene from the given profiles DataFrame.

    Parameters:
    - number_of_genes (int): The number of genes to extract.
    - profiles (DataFrame): The DataFrame containing gene profiles.
    - index (int): The starting index of the gene to extract.

    Returns:
    - DataFrame: A DataFrame containing the extracted gene profiles.
    """
    return profiles.iloc[:, index:index+number_of_genes]

#####################################################
# get_single_gene_duplicates
#####################################################   
def get_single_gene_duplicates(number_of_genes, X_profiles, index, duplicates_number):
    """
    Generate duplicates of a single gene's profile.

    Parameters:
    - number_of_genes (int): The total number of genes.
    - X_profiles (DataFrame): The DataFrame containing gene profiles.
    - index (int): The index of the gene to duplicate.
    - duplicates_number (int): The number of duplicates to generate.

    Returns:
    - result_df (DataFrame): The DataFrame containing the duplicated gene profiles.
    """
    original_df = get_single_gene(number_of_genes, X_profiles, index)
    # Create 40 copies of the original DataFrame
    copies = [original_df.copy() for _ in range(duplicates_number)]

    # Concatenate the copies along axis 1
    result_df = pd.concat(copies, axis=1)

    # Another Method
    # Convert the original DataFrame to a NumPy array
    # original_array = X_profiles.to_numpy()

    # # Create an array containing 40 copies of the original array
    # copies_array = np.tile(original_array, (1, 40))

    # Reshape the array to concatenate along axis 1
    return  result_df


#####################################################
# generate_permutations
#####################################################   
def generate_permutations():
    """
    Generate all possible permutations of a list of elements.

    Returns:
        permutations_list (list): A list of lists, where each inner list represents a permutation.
    """
    elements = [0, 1, 2, 3, 4]
    permutations_list = list(permutations(elements))
    # Convert tuples to lists
    permutations_list = [list(permutation) for permutation in permutations_list]
    return permutations_list

#####################################################
# single_gene_permutation_creation
#####################################################   
def single_gene_permutation_creation(single_gene, result_permutations):
    """
    Create permutations of a single gene and return the resulting gene image dataframe.

    Parameters:
    single_gene (DataFrame): A pandas DataFrame representing a single gene.
    result_permutations (list): A list of permutations to be applied to the gene.

    Returns:
    gene_image_df (ndarray): A numpy array representing the gene image dataframe.

    """
    one_new_gene_with_all_permutations = []
    np_single_gene = single_gene.values

    for permutation in result_permutations:
        one_new_gene_with_all_permutations.append(np_single_gene[:, permutation])

    gene_image_df = np.concatenate(one_new_gene_with_all_permutations, axis=1)
    return gene_image_df



#####################################################
# create_gene_image_dataset_with_permutation
#####################################################   
def create_gene_image_dataset_with_permutation(number_of_genes, profiles, result_permutations):
    """
    Creates a gene image dataset with permutations.

    Parameters:
    - number_of_genes (int): The number of genes per image.
    - profiles (numpy.ndarray): The gene profiles.
    - result_permutations (numpy.ndarray): The permutations to apply to each gene.

    Returns:
    - numpy.ndarray: The gene image dataset with permutations.
    """
    images = []
    for gene_i in tqdm(range(int(profiles.shape[1] / number_of_genes))):
        gene_image = single_gene_permutation_creation(get_single_gene(number_of_genes, profiles, gene_i * number_of_genes), result_permutations)
        images.append(gene_image)
    return np.array(images)


#####################################################
# create_gene_image_dataset_duplicates
#####################################################
def create_gene_image_dataset_duplicates(number_of_genes, profiles, duplicates_number, tqdm_print = False):
    """
    Creates a dataset of gene images with duplicates.

    Args:
        number_of_genes (int): The number of genes per image.
        profiles (numpy.ndarray): The gene profiles.
        duplicates_number (int): The number of duplicates per gene image.
        tqdm_print (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        numpy.ndarray: The dataset of gene images with duplicates.
    """
    images = []
    if tqdm_print:
        for gene_i in tqdm(range(int(profiles.shape[1]/ number_of_genes))):
            gene_image = get_single_gene_duplicates(number_of_genes, profiles, gene_i * number_of_genes, duplicates_number)
            images.append(gene_image)
    else:
        for gene_i in (range(int(profiles.shape[1]/ number_of_genes))):
            gene_image = get_single_gene_duplicates(number_of_genes, profiles, gene_i * number_of_genes, duplicates_number)
            images.append(gene_image)
    
    # print('Gene shape', np.array(images).shape)
    # pdb.set_trace()
    return np.array(images)


#####################################################
# create_gene_image_dataset_single_gene
#####################################################
def create_gene_image_dataset_single_gene(number_of_genes, profiles):
    """
    Create a dataset of gene images for a single gene.

    Args:
        number_of_genes (int): The number of genes per image.
        profiles (numpy.ndarray): The gene profiles.

    Returns:
        numpy.ndarray: An array of gene images.

    """
    images = []
    for gene_i in tqdm(range(int(profiles.shape[1]/ number_of_genes))):
        gene_image = get_single_gene(profiles, gene_i * number_of_genes)
        images.append(gene_image)
    return np.array(images)
