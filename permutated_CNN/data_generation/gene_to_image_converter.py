import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_gene(gene):
    # Set the figure size
    fig, ax = plt.subplots(figsize=(80, 80))  # You can adjust the values (width, height) in inches

    # Display the image using Matplotlib
    ax.imshow(gene, cmap='gray')  # 'gray' colormap for grayscale images
    ax.set_title('Image from NumPy Array')
    # Show the plot
    plt.show()

    
def get_single_gene(profiles, index):
    return profiles.iloc[:, index:index+5]

def get_single_gene_duplicates(X_profiles, index, duplicates_number):
    original_df = get_single_gene(X_profiles, index)
    # Create 40 copies of the original DataFrame
    copies = [original_df.copy() for _ in range(duplicates_number)]

    # Concatenate the copies along axis 1
    result_df = pd.concat(copies, axis=1)

    # # Convert the original DataFrame to a NumPy array
    # original_array = X_profiles.to_numpy()

    # # Create an array containing 40 copies of the original array
    # copies_array = np.tile(original_array, (1, 40))

    # Reshape the array to concatenate along axis 1
    return  result_df

from itertools import permutations

def generate_permutations():
    elements = [0, 1, 2, 3, 4]
    permutations_list = list(permutations(elements))

    # Convert tuples to lists
    permutations_list = [list(permutation) for permutation in permutations_list]

    return permutations_list

def single_gene_permutation_creation(single_gene, result_permutations):
    one_new_gene_with_all_permutations = []
    for i in (range(len(result_permutations))):
        one_new_gene_with_all_permutations.append(single_gene.iloc[:, result_permutations[i]])
    gene_image_df = pd.concat(one_new_gene_with_all_permutations, axis=1)
    return gene_image_df




def create_gene_image_dataset_with_permutation(profiles, result_permutations):
    images = []
    for gene_i in tqdm(range(int(profiles.shape[1]/ 5))):
        gene_image = single_gene_permutation_creation(get_single_gene(profiles, gene_i * 5) , result_permutations)
        images.append(gene_image.values)
    return np.array(images)


def create_gene_image_dataset_duplicates(profiles, duplicates_number):
    images = []
    for gene_i in tqdm(range(int(profiles.shape[1]/ 5))):
        gene_image = get_single_gene_duplicates(profiles, gene_i * 5, duplicates_number)
        images.append(gene_image)
    return np.array(images)

def create_gene_image_dataset_single_gene(profiles):
    images = []
    for gene_i in tqdm(range(int(profiles.shape[1]/ 5))):
        gene_image = get_single_gene(profiles, gene_i * 5)
        images.append(gene_image)
    return np.array(images)
