
from permutated_CNN.model.CNN_structure import *

import argparse
## Pytorch Version
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc

# Function to extract the numeric part of the filename
def extract_numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

def get_gene_images_files_list(data_dir, file_extension=".npy"):
        # Get a list of all files in the directory
    files = os.listdir(data_dir)

    # Filter files to include only those that match the pattern
    files = [file for file in files if file.endswith(file_extension)]

    # Sort the files based on their numeric values
    files.sort(key=extract_numeric_part)
    return files

def main():


    # Get arguments 
    parser = argparse.ArgumentParser(description="Train a CNN model for regression")
    parser.add_argument("--gene_data_dir", type=str, help="Directory containing the gene images")
    parser.add_argument("--rates_data_dir", type=str, help="Directory containing the gene rates")
    parser.add_argument("--model_dir", type=str, help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    gene_data_dir = args.gene_data_dir
    rates_data_dir = args.rates_data_dir
    model_dir = args.model_dir
    epochs = args.epochs
    batch_size = args.batch_size


    # Load the data
    gene_profiles_files = get_gene_images_files_list(gene_data_dir)
    rates = pd.read_csv(rates_data_dir)
    rates = rates.iloc[:, 10:]
    ## Create Permutated CNN model

    # Specify the input shape based on your data (e.g., image dimensions)
    input_shape = (1, 100, 600)  # Channels, Height, Width
    num_outputs = 10  # Regression output

    # Create the model
    model = RegressionCNN(input_shape, num_outputs)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Training the model

    for gene_images_file in gene_profiles_files:
        gene_image_df = np.load(gene_images_file)
        print('gene_image_df.shape:', gene_image_df.shape)
        

    
















# Running 

if __name__ == "__main__":
    main()

