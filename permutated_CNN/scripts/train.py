
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
from sklearn.model_selection import train_test_split
import copy
from tensorboardX import SummaryWriter



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

def main():
    # Get arguments 
    parser = argparse.ArgumentParser(description="Train a CNN model for regression")
    parser.add_argument("--main_dir", type=str, help="Directory containing the required files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_outputs", type=int, default=10, help="Number of outputs for the regression model")
    parser.add_argument("--sub_training_batch", type=int, default=10, help="Number of sub-training batches")
    parser.add_argument('--input_gene_image_size', metavar='N', type=str, help='Three integers for the input gene image size in double quotation mark. Ex: \"1, 100, 600\"')
    parser.add_argument("--gene_image_type", type=int, default=0, help="gene image type, 0 for permutated gene images, 1 for duplicates gene images")
    args = parser.parse_args()

    directory = args.main_dir
    gene_image_type = args.gene_image_type
    if gene_image_type == 0:
        gene_data_dir = f"{directory}/gene_images/permutated_gene_images/"
    else:
        gene_data_dir = f"{directory}/gene_images/duplicates_gene_images/"
        
    rates_data_dir = f"{directory}/concatenated_data/"
    model_dir = f"{directory}/model_checkpoints/"
    log_dir = f"{directory}/logs/"
    num_outputs = args.num_outputs

    num_epochs = args.epochs
    sub_training_batch = args.sub_training_batch
    batch_size = args.batch_size

    input_shape_str = args.input_gene_image_size.split(',')
    input_shape = tuple(map(int, input_shape_str))

    print("input_shape:", input_shape)

    # Load the data
    gene_profiles_files = get_files_list(gene_data_dir, file_extension=".npy")
    gene_rates_files = get_files_list(rates_data_dir, file_extension=".csv")

    print("gene_profiles_files:", gene_profiles_files)
    print("gene_rates_files:", gene_rates_files)

    ## Create Permutated CNN model

    # Create the model
    model = RegressionCNN(input_shape, num_outputs)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # save for the best agent
    best_MSE = 100000
    best_model = copy.deepcopy(model)
    early_stopping = 35

    # Training the model

    log_trending = f'{log_dir}/runs/'


    try:
        print("Create log directory")
        os.mkdir(log_trending)
    except:
        print("Directory already exists")

    try:
         print("Create model directory")
         os.mkdir(model_dir)
    except:
        print("Directory already exists")


    _summ_writer = SummaryWriter(log_trending, flush_secs=1, max_queue=1)

    # For tensorboard logger
    runs_tracking = 0


    print("Load Validation and Test Data")
    # use the last piece of data for validation, and testing
    X = np.load(gene_data_dir + gene_profiles_files[-1])[:, None, :, :]
    y = pd.read_csv(rates_data_dir + gene_rates_files[-1]).iloc[:, -num_outputs:].values
    
    delet_x, X, delet_y, y = train_test_split(X, y, test_size=0.1, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Free memory space
    del X, y
    del delet_x, delet_y

    # save as pytorch data loader
    val_dataset = RegressionDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = RegressionDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    print("Load Success!")
    print(f"X_val.shape: {X_val.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_val.shape: {y_val.shape}")
    print(f"y_test.shape: {y_test.shape}")


    Early_stopping_MSE = 1000000

    for Gen in range(num_epochs):

        whole_run_loss = 0.0

        # for file_id in range(len(gene_profiles_files)):
        for file_id in range(len(gene_profiles_files) - 1):

            # For tensorboard logger
            runs_tracking += 1
            print("####################################################################")
            print(f"Train {file_id + 1} th batch")

            gene_image_df = np.load(gene_data_dir + gene_profiles_files[file_id])
            rates = pd.read_csv(rates_data_dir + gene_rates_files[file_id])

            rates = rates.iloc[:, -num_outputs:]

            print('rates.shape:', rates.shape)
            print('gene_image_df.shape:', gene_image_df.shape)

            # table_columns = rates.columns

            X_train = gene_image_df[:, None, :, :] 
            y_train = rates.values

            # Split the dataset into training and validation sets (80% training, 20% validation)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Create a custom dataset
                    
            # Create the RegressionDataset instances for training and validation
            train_dataset = RegressionDataset(X_train, y_train)

            # Create the DataLoader for training and validation
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Training the model
            batch_val_loss = 0.0

            # Free memory space
            del gene_image_df

            for epoch in range(sub_training_batch):
                # Training
                model.train()
                num_batches = len(train_dataloader)
                progress_bar = tqdm(total=num_batches, desc='Training')
                training_loss = 0.0 
                for inputs, targets in train_dataloader:
                    # Transfer data to GPU
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    # Update the tqdm progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item())
                
                # Close the tqdm progress bar
                progress_bar.close()
                training_loss /= len(train_dataloader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        # Transfer validation data to GPU
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = model(val_inputs)
                        val_saving = pd.DataFrame((val_outputs.cpu().numpy()))
                        val_loss += criterion(val_outputs, val_targets).item()
                        
                # Average validation loss
                val_loss /= len(val_dataloader)
                batch_val_loss += val_loss

                # log trending 
                # _summ_writer.add_scalar('{}'.format('Val_Loss'), val_loss, epoch)
                # _summ_writer.add_scalar('{}'.format('Training_Loss'), training_loss, epoch) 

                _summ_writer.add_scalars('{}'.format('batch_loss'), {
                        'Val_Loss': val_loss,
                        'Training_Loss': training_loss,
                }, runs_tracking)

                if val_loss <= best_MSE:
                    print("####################################################################")
                    print(f'Find new batch best model at {val_loss}')
                    print("####################################################################")
                    batch_best_model = copy.deepcopy(model)
                    # Save the best model
                    torch.save(batch_best_model.state_dict(), f"{model_dir}/batch_best_model.pth")
                    best_MSE = val_loss

                print(f'Gen {Gen} Batch Epoch {epoch+1}/{sub_training_batch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

                if early_stopping < 0:
                    print("####################################################################")
                    print("Stop the training by early stopping")
                    print("####################################################################")
                    break

            # end batch for loop
            batch_val_loss /= sub_training_batch
            whole_run_loss += batch_val_loss

        # end one single training loop
        whole_run_loss /= (len(gene_profiles_files) - 1)

        if whole_run_loss <= Early_stopping_MSE:
            print("####################################################################")
            print(f'Find new whole run best model at {whole_run_loss}')
            print("####################################################################")
            best_model = copy.deepcopy(model)
            # Save the best model
            torch.save(best_model.state_dict(), f"{model_dir}/whole_run_best_model.pth")
            Early_stopping_MSE = whole_run_loss
        else:
            early_stopping -= 1








    
















# Running 

if __name__ == "__main__":
    main()

