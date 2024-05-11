import time
from permutated_CNN.model.CNN_structure import *
from permutated_CNN.model.ResNet_structure import *

import argparse
## Pytorch Version
import pandas as pd 
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import copy
from tensorboardX import SummaryWriter
import pyfiglet
from datetime import datetime
import subprocess


def np_load_genome_data(file_path, chunks=None):
    blocksize = 1000  # tune this for performance/granularity
    try:
        mmap = np.load(file_path, mmap_mode='r')
        y = np.empty_like(mmap)
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        if chunks is None:
            chunks = n_blocks
        else:
            chunks = min(chunks, n_blocks)
        for b in tqdm(range(chunks)):
            y[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again
    return y


# Function to generate ASCII art using the DOOM font
def generate_doom_ascii(text):
        try:
            # Create a Figlet font object with the DOOM font style
            doom_font = pyfiglet.Figlet(font='doom')

            # Generate the ASCII art from the input text using the DOOM font
            ascii_art = doom_font.renderText(text)

            return ascii_art

        except pyfiglet.FontNotFound:
            return "DOOM font not found. Please make sure you have the DOOM font installed."
        

# Function to create a directory
def create_folders(directory, name):
    try:
        print(f"Create {name} directory")
        os.mkdir(directory)
        print(f"{name} directory created")
    except:
        print("Directory already exists")

# Function to extract the numeric part of the filename
def extract_numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

def get_files_list(data_dir, file_extension=".npy"):
        # Get a list of all files in the directory
    files = os.listdir(data_dir)

    # Filter files to include only those that match the pattern
    files = [file for file in files if file.endswith(file_extension)]

    # Sort the files based on their numeric values
    files.sort(key=extract_numeric_part)
    return files


def read_csv_with_chunksize(file_path, chunksize = 8000):
    # Get the total number of lines in the file
    print("Read CSV file...")
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




def main():
    # Get arguments 
    print("####################################################################")
    print("Model Prediction Start!")
    parser = argparse.ArgumentParser(description="Train a CNN model for regression")
    parser.add_argument("--main_dir", type=str, help="Directory containing the required files")
    parser.add_argument("--model_checkpoint_path", type=str, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_outputs", type=int, default=10, help="Number of outputs for the regression model")
    parser.add_argument("--gene_image_path", type=str, help="gene image path")
    parser.add_argument("--gene_rates_path", type=str, help="gene rates path")
    parser.add_argument("--ResNet_depth", type=int, default=50, help="ResNet depth")
    parser.add_argument("--image_length", type=int, default=50, help="Image_length")
    parser.add_argument("--padding_length", type=int, default=50, help="padding_length")


    
    args = parser.parse_args()

    padding_length = args.padding_length
    directory = args.main_dir
    create_folders(f"{directory}/predictions", 'predictions')
    model_checkpoint_path = args.model_checkpoint_path

    gene_image_path = args.gene_image_path
    gene_rates_path = args.gene_rates_path
    ResNet_depth = args.ResNet_depth

    num_outputs = args.num_outputs
    batch_size = args.batch_size
    image_length = args.image_length


    print("LOAD RESNET MODEL")
    model_parameters={}
    model_parameters['resnet18'] = ([64,128,256,512],[2,2,2,2],1,False)
    model_parameters['resnet34'] = ([64,128,256,512],[3,4,6,3],1,False)
    model_parameters['resnet50'] = ([64,128,256,512],[3,4,6,3],4,True)
    model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
    model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)
    
    if ResNet_depth == 18:
            architecture_setting = model_parameters['resnet18'] 
    elif ResNet_depth == 34:
        architecture_setting = model_parameters['resnet34'] 
    elif ResNet_depth == 50:
        architecture_setting = model_parameters['resnet50'] 
    elif ResNet_depth == 101:
        architecture_setting = model_parameters['resnet101']
    elif ResNet_depth == 152:
        architecture_setting = model_parameters['resnet152'] 

    # Create the model
    print("architecture_setting:", architecture_setting)
    model = ResNet(architecture_setting, in_channels=1, num_classes = num_outputs)

     # Load the state dictionary from the checkpoint file
    checkpoint = torch.load(model_checkpoint_path)

    # Load the state dictionary into your model
    model.load_state_dict(checkpoint)
    print("####################################################################")
    print(generate_doom_ascii(f"Model  checkpoint Load  Success!"))
    print("--------------------------------------------------------------------")
    print(f"Load model checkpoint from {model_checkpoint_path}")
    print("####################################################################")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr = learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("####################################################################")
    print(f"Model is using {device} for training")
    print("####################################################################")
    
    model.to(device)

    print(model)


    # Load the gene image data
    X = np_load_genome_data(gene_image_path)
    print("X.shape:", X.shape)
    y = read_csv_with_chunksize(gene_rates_path)
    y = y.iloc[:, -num_outputs:].values
    print("y.shape:", y.shape)

    # Pad the array to shape (n, 1000, 100) with a constant value of 0.5
    if X.shape[1] < padding_length:
        print("Padding the array to 1000 tips for predicting")
        X = np.pad(X, ((0, 0), (0, padding_length - X.shape[1]), (0, 0)), constant_values=0.5)

    # only take 6000 samples for validation and testing
    X = X[:, None, :, :image_length]
    # X = X[:8000]
    # y = y[:8000]

    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    val_dataset = RegressionDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # X_truncated_image = X_val.copy()
    # X_truncated_image[400:, 0, :, :] = 0.5

    # # np.set_printoptions(threshold=np.inf)
    # # print(X_truncated_image[0])
    # print("X_val.shape:", X_val.shape)
    # print("X_truncated_image.shape:", X_truncated_image.shape)
    # X_truncated_val_dataset = RegressionDataset(X_truncated_image, y_val)
    # X_truncated_val_dataloader = DataLoader(X_truncated_val_dataset, batch_size=batch_size, shuffle=False)



    
    print("Load Success!")
    print(f"X_val.shape: {X_val.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_val.shape: {y_val.shape}")
    print(f"y_test.shape: {y_test.shape}")

    pd.DataFrame(y_val).to_csv(f"{directory}/predictions/true_values.csv", index=False)
    

    del X, y, val_dataset,  X_val, y_val, X_test, y_test

    print("####################################################################")
    print("Original Image Profile Validation Loss")
    val_loss = 0
    num_batches = len(val_dataloader)
    progress_bar = tqdm(total=num_batches, desc='Validation')

    predictions = []
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            # Transfer validation data to GPU
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_saving = pd.DataFrame((val_outputs.cpu().numpy()))
            predictions.append(val_saving)
            val_loss_criterion = criterion(val_outputs, val_targets)
            val_loss += val_loss_criterion.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=val_loss_criterion.item())
    print("Validation MSE Loss: ", val_loss / len(val_dataloader))
    print("####################################################################\n\n")


    print("Save predictions")
    combined_df = pd.concat(predictions, ignore_index=True)
    combined_df.to_csv(f"{directory}/predictions/predictions.csv", index=False)
   
    # print("####################################################################")
    # print("Truncated Image Profile Validation Loss")
    # val_loss = 0
    # num_batches = len(val_dataloader)
    # progress_bar = tqdm(total=num_batches, desc='Validation')

    # with torch.no_grad():
    #     for val_inputs, val_targets in X_truncated_val_dataloader:
    #         # Transfer validation data to GPU
    #         val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
    #         val_outputs = model(val_inputs)
    #         val_saving = pd.DataFrame((val_outputs.cpu().numpy()))
    #         val_loss_criterion = criterion(val_outputs, val_targets)
    #         val_loss += val_loss_criterion.item()
    #         progress_bar.update(1)
    #         progress_bar.set_postfix(loss=val_loss_criterion.item())
    # print("\nValidation MSE Loss: ", val_loss / len(val_dataloader))

    print("\n\n\n")


# Running 

if __name__ == "__main__":
    main()
