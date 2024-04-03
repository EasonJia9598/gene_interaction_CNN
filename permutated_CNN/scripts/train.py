
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

# # Show progress bar when loading numpy data
# def np_load_genome_data(file_path):
#     blocksize = 5000  # tune this for performance/granularity
#     try:
#         mmap = np.load(file_path, mmap_mode='r')
#         y = np.empty_like(mmap)
#         n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
#         for b in tqdm(range(n_blocks)):
#             # print('progress: {}/{}'.format(b, n_blocks))  # use any progress indicator
#             y[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
#     finally:
#         del mmap  # make sure file is closed again
#     return y

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
    parser = argparse.ArgumentParser(description="Train a CNN model for regression")
    parser.add_argument("--main_dir", type=str, help="Directory containing the required files")
    parser.add_argument("--if_use_temperary_ssd", type=int, help="If use temperary SSD, 0 for without using, 1 for using")
    parser.add_argument("--temperary_ssd_dr", type=str, help="Directory saving the loading file")
    parser.add_argument("--load_model_checkpoint", type=int, help="If Load the model checkpoint, 0 for without loading, 1 for loading the model checkpoint")
    parser.add_argument("--model_checkpoint_path", type=str, help="Model checkpoint path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_outputs", type=int, default=10, help="Number of outputs for the regression model")
    parser.add_argument("--sub_training_batch", type=int, default=10, help="Number of sub-training batches")
    parser.add_argument('--input_gene_image_size', metavar='N', type=str, help='Three integers for the input gene image size in double quotation mark. Ex: \"1, 100, 600\"')
    parser.add_argument("--gene_image_type", type=int, default=0, help="gene image type, 0 for permutated gene images, 1 for duplicates gene images")
    parser.add_argument("--log_file_name", type=str, default="Permutated_CNN", help="Log file name")
    parser.add_argument("--model_type", type=str, default="CNN", help="Model type, CNN or ResNet")
    parser.add_argument("--ResNet_depth", type=int, default=50, help="ResNet depth")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--training_data_size", type=int, default=40000, help="Training data size")
    args = parser.parse_args()

    model_type = args.model_type
    directory = args.main_dir
    gene_image_type = args.gene_image_type
    ResNet_depth = args.ResNet_depth
    learning_rate = args.learning_rate
    load_model_checkpoint = args.load_model_checkpoint
    model_checkpoint_path = args.model_checkpoint_path
    if_use_temperary_ssd = int(args.if_use_temperary_ssd)





    if gene_image_type == 0:
        gene_data_dir = f"{directory}/gene_images/permutated_gene_images/"
    else:
        gene_data_dir = f"{directory}/gene_images/duplicates_gene_images/"

    rates_data_dir = f"{directory}/rates/"

    log_file_name = args.log_file_name


    temperary_ssd_dr = args.temperary_ssd_dr



    current_time = datetime.now()
    formatted_time = current_time.strftime("%m_%d_%H_%M_%S")
    
    log_dir = f"{directory}/logs/{formatted_time}_{log_file_name}"
    model_dir = f"{directory}/model_checkpoints/{formatted_time}_{log_file_name}"
    
    if if_use_temperary_ssd == 1:
        create_folders(temperary_ssd_dr, "temperary_ssd_dr")

    create_folders(f"{directory}/logs", "main log")
    create_folders(f"{directory}/model_checkpoints", "model checkpoints")

    create_folders(log_dir, "current log")
    create_folders(model_dir, "current model")


    if if_use_temperary_ssd == 1:
        temperary_ssd_dr_model_checkpoints = f"{temperary_ssd_dr}/model_checkpoints/{formatted_time}_{log_file_name}"
        create_folders(f"{temperary_ssd_dr}/model_checkpoints/", 'temperary_ssd_dr_main_model_checkpoints')
        create_folders(temperary_ssd_dr_model_checkpoints, 'temperary_ssd_dr_model_checkpoints')
    
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

    
    if model_type == 'CNN':
        print("LOAD CNN MODEL")
        # Create the model
        model = RegressionCNN(input_shape, num_outputs)
    else:
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

    if load_model_checkpoint:
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
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    # Note: PyTorch does not have a direct equivalent to model.summary() in Keras/TensorFlow.
    # You can print the model to see its architecture.
    # Move model to GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("####################################################################")
    print(f"Model is using {device} for training")
    print("####################################################################")
    
    model.to(device)

    print(model)

    print(optimizer)

    # save for the best agent
    best_MSE = 100000
    best_model = copy.deepcopy(model)
    early_stopping = 35

    # Training the model

    log_trending = f'{log_dir}/runs/'


    create_folders(log_trending, "log trending")

    _summ_writer = SummaryWriter(log_trending, flush_secs=1, max_queue=1)

    print("Genes file: ", gene_profiles_files[-1])


    if if_use_temperary_ssd == 1:
        # In order to maxiumn the speed of SSD loading, we will load the first batch of data to the SSD
        first_atch_file = f"{temperary_ssd_dr}/{gene_profiles_files[0]}"
        process_wait_trigger = False
        if os.path.exists(first_atch_file):
            print("First batch Data File exists")
        else:
            print("Copy First Batch Data to SSD")
            process_wait_trigger = True
            # Define the terminal command you want to run
            terminal_command = "cp " + gene_data_dir + gene_profiles_files[0] + " " +   temperary_ssd_dr 
            # Execute the terminal command in the background
            process = subprocess.Popen(terminal_command, shell=True)

        validation_test_data_path = temperary_ssd_dr + '/' + gene_profiles_files[-1]

        if os.path.exists(validation_test_data_path):
            print("Validation Data File exists")
        else:
            process_wait_trigger = True
            print("Copy Validation Batch Data to SSD")
            # Define the terminal command you want to run
            terminal_command = "cp " + gene_data_dir + gene_profiles_files[-1] + " " +   temperary_ssd_dr 
            # Execute the terminal command in the background
            process_val = subprocess.Popen(terminal_command, shell=True)
            print("####################################################################")
            print("Please wait for the first batch data to be loaded to the SSD")
            print("####################################################################")

        if process_wait_trigger:
            process.wait()
            process_val.wait()


        print("Load Validation and Test Data")
        # use the last piece of data for validation, and testing
        # X = np_load_genome_data(gene_data_dir + gene_profiles_files[-1])
        X = np_load_genome_data(temperary_ssd_dr + '/' + gene_profiles_files[-1])
    else:
        X = np_load_genome_data(gene_data_dir + gene_profiles_files[-1], 12)

    X = X[:, None, :, :]
    print("X.shape:", X.shape)
    print("Rates file: ", gene_rates_files[-1])
    y = read_csv_with_chunksize(rates_data_dir + gene_rates_files[-1])
    y = y.iloc[:, -num_outputs:].values
    print("y.shape:", y.shape)
    print("Success!")
    
    # only take 6000 samples for validation and testing
    X = X[:10000]
    y = y[:10000]

    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Free memory space
    del X, y
    # del delet_x, delet_y

    # save as pytorch data loader
    val_dataset = RegressionDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    del val_dataset

    # test_dataset = RegressionDataset(X_test, y_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    print("Load Success!")
    print(f"X_val.shape: {X_val.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_val.shape: {y_val.shape}")
    print(f"y_test.shape: {y_test.shape}")

    del X_val, y_val, X_test, y_test

    # # Temperoray code for checking
    # network_type = 'star'
    # print(network_type.upper())
    # # X_test = np.load(f'/home/eeepc/Documents/image_genes/test_data/permutated_gene_image_{network_type}.npy')
    # X_test = np.load(f'/home/eeepc/Documents/image_genes/test_data/{network_type}_profile_image.npy')
    # X_test = X_test[:, None, :, :]
    # y_test = pd.read_csv(f'/home/eeepc/Documents/image_genes/test_data/{network_type}_rates_data.csv')
    # # print(y_test.value_counts())
    # y_test = y_test.iloc[:, 10:].values



    # test_dataset = RegressionDataset(X_test, y_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



    
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

            if if_use_temperary_ssd == 1:
                if file_id > 1:
                    # Define the terminal command you want to run
                    terminal_command = "rm " + temperary_ssd_dr + '/' + gene_profiles_files[file_id - 1]
                    # Execute the terminal command in the background
                    rm_process = subprocess.Popen(terminal_command, shell=True)


                # Define the terminal command you want to run
                # Preload the data file to SSD
                terminal_command = "cp " + gene_data_dir + gene_profiles_files[(file_id + 1) % (len(gene_profiles_files) - 1)] + " " +   temperary_ssd_dr 
                # Execute the terminal command in the background
                process = subprocess.Popen(terminal_command, shell=True)


                condition_to_move_on = True

                while condition_to_move_on:
                    try:
                        gene_image_df = np_load_genome_data(temperary_ssd_dr + '/' + gene_profiles_files[file_id])
                    finally:
                        condition_to_move_on = False

            else:
                gene_image_df = np_load_genome_data(gene_data_dir + gene_profiles_files[file_id], int(args.training_data_size / 1000))


            print(gene_profiles_files[file_id])
            rates = read_csv_with_chunksize(rates_data_dir + gene_rates_files[file_id])
            rates = rates.iloc[:, -num_outputs:]
            print(gene_rates_files[file_id])


            # table_columns = rates.columns

            X_train = gene_image_df[:args.training_data_size, None, :, :] 
            y_train = rates[:args.training_data_size].values
            
            print('rates.shape:', y_train.shape)
            print('gene_image_df.shape:', X_train.shape)

            # Free memory space
            del gene_image_df
            del rates
            
            # Split the dataset into training and validation sets (80% training, 20% validation)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Create a custom dataset
                    
            # Create the RegressionDataset instances for training and validation
            train_dataset = RegressionDataset(X_train, y_train)

            del X_train, y_train

            # Create the DataLoader for training and validation
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Training the model
            batch_val_loss = 0.0

            del train_dataset

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
                num_batches = len(val_dataloader)
                progress_bar = tqdm(total=num_batches, desc='Validation')

                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        # Transfer validation data to GPU
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = model(val_inputs)
                        val_saving = pd.DataFrame((val_outputs.cpu().numpy()))
                        val_loss_criterion = criterion(val_outputs, val_targets)
                        val_loss += val_loss_criterion.item()
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=val_loss_criterion.item())
                        
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
                    print("\n####################################################################")
                    print(f'Find new batch best model at {val_loss}')
                    print("####################################################################")
                    batch_best_model = copy.deepcopy(model)
                    # Save the best model
                    # torch.save(batch_best_model.state_dict(), f"{model_dir}/batch_best_model_{np.round(val_loss,3)}.pth")
                    if if_use_temperary_ssd == 1:
                        torch.save(batch_best_model.state_dict(), f"{temperary_ssd_dr_model_checkpoints}/batch_best_model_{np.round(val_loss,3)}.pth")
                    else:
                        torch.save(batch_best_model.state_dict(), f"{model_dir}/batch_best_model_{np.round(val_loss,3)}.pth")
                    best_MSE = val_loss

                print(f'Gen {Gen}: Batch {file_id + 1} {epoch+1}/{sub_training_batch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

                
                # prediction_list = []
                # truth_list = []

                # test_loss = 0

                # with torch.no_grad():
                #     for test_inputs, test_targets in test_dataloader:
                #         # Transfer validation data to GPU
                #         test_inputs = test_inputs.to(device)
                #         prediction_tf = model.forward(test_inputs)
                #         test_loss += criterion(prediction_tf.detach().cpu(), test_targets).item()
                #         predictions = pd.DataFrame(prediction_tf.detach().cpu().numpy())
                #         prediction_list.append(predictions)
                #         truth_list.append(pd.DataFrame(test_targets.numpy()))
                        

                # predictions = pd.concat(prediction_list, axis=0)
                # truth = pd.concat(truth_list, axis=0)
                # print(f"STAR test set LOSS {(predictions.values - truth.values).mean()}")


            # end batch for loop
            batch_val_loss /= sub_training_batch
            whole_run_loss += batch_val_loss
            print("Wait for images copy to finish")

            if if_use_temperary_ssd == 1:
                process.wait()
                if file_id > 1:
                    rm_process.wait()
                print("Success!")

        # end one single training loop
        whole_run_loss /= (len(gene_profiles_files) - 1)

        del train_dataloader


        if whole_run_loss <= Early_stopping_MSE:
            print("####################################################################")
            print(f'Find new whole run best model at {whole_run_loss}')
            print(generate_doom_ascii(f'Find new whole run best model at {np.round(whole_run_loss,2)}'))
            print("####################################################################")
            best_model = copy.deepcopy(model)
            # Save the best model
            # torch.save(best_model.state_dict(), f"{model_dir}/whole_run_best_model_MSE_{np.round(whole_run_loss,2)}.pth")
            if if_use_temperary_ssd == 1:
                torch.save(best_model.state_dict(), f"{temperary_ssd_dr_model_checkpoints}/whole_run_best_model_MSE_{np.round(whole_run_loss,2)}.pth")
            else:
                torch.save(best_model.state_dict(), f"{model_dir}/whole_run_best_model_MSE_{np.round(whole_run_loss,2)}.pth")

            Early_stopping_MSE = whole_run_loss
        else:
            print("####################################################################")
            print(f'No improvement at {np.round(whole_run_loss,2)}')
            print("####################################################################")
            early_stopping -= 1
            
        if early_stopping < 0:
            print("####################################################################")
            print("Stop the training by early stopping")
            print("####################################################################")
            break


# Running 

if __name__ == "__main__":
    main()

