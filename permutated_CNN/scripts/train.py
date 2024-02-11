
from permutated_CNN.model.CNN_structure import *
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

def np_load_genome_data(file_path):
    blocksize = 1024  # tune this for performance/granularity
    try:
        mmap = np.load(file_path, mmap_mode='r')
        y = np.empty_like(mmap)
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        for b in tqdm(range(n_blocks)):
            # print('progress: {}/{}'.format(b, n_blocks))  # use any progress indicator
            y[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again
    return y

def generate_doom_ascii(text):
        try:
            # Create a Figlet font object with the DOOM font style
            doom_font = pyfiglet.Figlet(font='doom')

            # Generate the ASCII art from the input text using the DOOM font
            ascii_art = doom_font.renderText(text)

            return ascii_art

        except pyfiglet.FontNotFound:
            return "DOOM font not found. Please make sure you have the DOOM font installed."
        


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
    parser.add_argument("--log_file_name", type=str, default="Permutated_CNN", help="Log file name")
    args = parser.parse_args()

    directory = args.main_dir
    gene_image_type = args.gene_image_type
    if gene_image_type == 0:
        gene_data_dir = f"{directory}/gene_images/permutated_gene_images/"
    else:
        gene_data_dir = f"{directory}/gene_images/duplicates_gene_images/"

    rates_data_dir = f"{directory}/rates/"

    log_file_name = args.log_file_name


    current_time = datetime.now()
    formatted_time = current_time.strftime("%m_%d_%H_%M_%S")
    
    log_dir = f"{directory}/logs/{formatted_time}_{log_file_name}"
    model_dir = f"{directory}/model_checkpoints/{formatted_time}_{log_file_name}"
    
    create_folders(f"{directory}/logs", "main log")
    create_folders(f"{directory}/model_checkpoints", "model checkpoints")

    create_folders(log_dir, "current log")
    create_folders(model_dir, "current model")
    
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


    create_folders(log_trending, "log trending")

    _summ_writer = SummaryWriter(log_trending, flush_secs=1, max_queue=1)


    print("Load Validation and Test Data")
    # use the last piece of data for validation, and testing
    X = np_load_genome_data(gene_data_dir + gene_profiles_files[-1])
    X = X[:, None, :, :]
    y = pd.read_csv(rates_data_dir + gene_rates_files[-1]).iloc[:, -num_outputs:].values
    
    # only take 6000 samples for validation and testing
    X = X[:6000]
    y = y[:6000]

    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Free memory space
    del X, y
    # del delet_x, delet_y

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

        print("####################################################################")
        print(f"Train {Gen} th epoch")
        print(generate_doom_ascii(f"Train No. {Gen} epoch"))
        print("####################################################################")
        whole_run_loss = 0.0

        # For tensorboard tracking
        runs_tracking = Gen * sub_training_batch

        # for file_id in range(len(gene_profiles_files)):
        for file_id in range(len(gene_profiles_files) - 1):
            print("####################################################################")
            print(f"Train No. {file_id + 1} batch over {len(gene_profiles_files) - 1} batches")
            print(generate_doom_ascii(f"Train No. {file_id + 1} batch"))
            print("####################################################################")

            # Load the data
            print("Load Batch Data")
            # gene_image_df = np.load(gene_data_dir + gene_profiles_files[file_id])
            
            gene_image_df = np_load_genome_data(gene_data_dir + gene_profiles_files[file_id])
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
                # For tensorboard logger
                runs_tracking += 1
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

                _summ_writer.add_scalars('{}'.format(f'batch_loss'), {
                        'Val_Loss': val_loss,
                        'Training_Loss': training_loss,
                }, runs_tracking)

                if val_loss <= best_MSE:
                    print("####################################################################")
                    print(f'Find new batch best model at {val_loss}')
                    print("####################################################################")
                    batch_best_model = copy.deepcopy(model)
                    # Save the best model
                    torch.save(batch_best_model.state_dict(), f"{model_dir}/batch_best_model_{np.round(val_loss,3)}.pth")
                    best_MSE = val_loss

                print(f'Gen {Gen}: Batch {file_id + 1} {epoch+1}/{sub_training_batch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

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
            print(generate_doom_ascii(f'Find new whole run best model at {np.round(whole_run_loss,2)}'))
            print("####################################################################")
            best_model = copy.deepcopy(model)
            # Save the best model
            torch.save(best_model.state_dict(), f"{model_dir}/whole_run_best_model_MSE_{np.round(whole_run_loss,2)}.pth")
            Early_stopping_MSE = whole_run_loss
        else:
            print("####################################################################")
            print(f'No improvement at {np.round(whole_run_loss,2)}')
            print("####################################################################")
            early_stopping -= 1



# Running 

if __name__ == "__main__":
    main()
