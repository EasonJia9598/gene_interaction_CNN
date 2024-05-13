# Implementation of my thesis: 
## Author: Zesheng Jia
## STUDY OF GENES INTERACTION RELATIONSHIPS USING REGRESSION CONVOLUTIONAL NEURAL NETWORKS WITH HYPOTHESIS TESTING ON LARGE-SCALE SELF-SIMULATED GENE PROFILES WITH EMBEDDED PHYLOGENETIC TREE STRUCTURES, Zesheng Jia

Link: https://www.mathstat.dal.ca/~tsusko/honours-theses/zesheng-jia.pdf

## Abstract: 
In this study, we evaluated multiple deep learning models, including a simple feedforward neural network, a Convolutional Neural Network (CNN), and the ResNet
with and without a bottleneck structure, for identifying gene interaction types using
5 genes profiles image that is embedded with phylogenetic tree structures. We combined these models with hypothesis testing, analyzed the ROC curve, and compared
the performance of the evolCCM and CNN models. A pipeline was developed to
determine the minimum dataset size for training, and model robustness was confirmed against genes’ order changes in the gene profile using the same trained model
checkpoint. The False Discovery Rate of the predictions was calibrated using the
Benjamini-Hochberg Procedure. Our findings indicated that the ResNet with a bottleneck structure and 34/50 layers yielded the smallest Mean Square Error and Root
Mean Square Error for predicting phylogenetic rates of 5 genes along with the training
time trade-off. The CNN model proved to be unaffected by changing the phylogenetic
tree structures across different data points during training and very little inference
time during predictions. We also conduct experiments when increase the number of
genes to 10 genes, 20 genes, and 40 genes. Future research will scale up training
data with more gene profiles. We hope this study can contribute to understanding
interaction types between genes pairs using deep learning models, large-scale data
simulation, and statistical analysis.

# Niagara Cluster Instructions
From SciNet Documentation: Installing your own Python Modules, https://docs.scinet.utoronto.ca/index.php/Installing_your_own_Python_Modules

In the terminal, first load a python module, e.g.

   ```
   module load NiaEnv/2019b python/3.11.5
   ```

Then create a directory for the virtual environments. One can put a virtual environment anywhere, but this directory structure is recommended:

   ```
   mkdir ~/.virtualenvs
   ```
Now we create our first virtualenv called myenv choose any name you like:

   ```
   virtualenv --system-site-packages ~/.virtualenvs/myenv
   ```
The "--system-site-packages" flag will use the system-installed versions of packages rather than installing them anew (the list of these packages can be found on the Python wiki page). This will result in fewer files created in your virtual environment. After that you can activate that virtual environment:

   ```
   source ~/.virtualenvs/myenv/bin/activate 
   ```
As you are in the virtualenv now, you can just type ```pip install <required module>``` to install any module into your virtual environment.

To go back to the normal python installation simply type

   ```
   deactivate
   ```

# Mist Cluster Instructions
> The Mist system is a cluster of 54 IBM servers each with 4 NVIDIA V100 “Volta” GPUs with 32 GB memory each, and with NVLINKs in between. Each node of the cluster has 256GB RAM. It has InfiniBand EDR interconnection providing GPU-Direct RMDA capability. This system is a combination of the GPU extension to the Niagara cluster and the refresh of the GPU cluster of the Southern Ontario Smart Computing Innovation Platform (SOSCIP). The Niagara GPU portion is available to Compute Canada users, while the SOSCIP portion will be used by allocated SOSCIP projects. By combining the resources, users from either group are able to take advantage of any unused computing resources of the other group. https://www.scinethpc.ca/mist/


## For install packages (customized for our usage)
Author: Zesheng Jia

```Shell
module load anaconda3
module load cuda/11.4.4
module load gcc/.core

conda create -n pytorch_env python=3.8
source activate pytorch_env

#must force to use Open-CE channel to avoid the cpu-only version of PyTorch from default Anaconda channel
conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/1.7.2/
conda config --set channel_priority strict

conda install -c https://ftp.osuosl.org/pub/open-ce/1.7.2/ pytorch=1.12.1 cudatoolkit=11.4

```

Logout and Restart the terminal

```Shell
source activate pytorch_env
```

Test if its installation is success:

```Shell
python3
```

```Python
import torch
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Install our Package

```
cd /******/zeshengj/gene_corelation_CNN/
pip install -r requirements.txt
pip install -e .

```

For Mist Cluster, we need to manually install any packages in requirements from conda.
By the following reasons.

> **Important note:** the majority of computer systems as of 2021 (laptops, desktops, and HPC) use the 64 bit x86 instruction set architecture (ISA) in their microprocessors produced by Intel and AMD. This ISA is incompatible with Mist, whose hardware uses the 64 bit PPC ISA (set to little endian mode). The practical meaning is that x86-compiled binaries (executables and libraries) cannot be installed on Mist. For this reason, the Niagara and Alliance (formerly Compute Canada) software stacks (modules) cannot be made available on Mist, and using closed-source software is only possible when the vendor provides a compatible version of their application. **Python applications** almost always rely on bindings to libraries originally written in C or C++, some of them are not available on PyPI or various Conda channels as precompiled binaries compatible with Mist. ***The recommended way to use Python on Mist is to create a [Conda](https://docs.scinet.utoronto.ca/index.php/Mist#Anaconda_.28Python.29) environment and install packages from the anaconda (default) channel, where most popular packages have a linux-ppc64le (Mist-compatible) version available.*** Some popular machine learning packages should be installed from the internal [Open-CE](https://docs.scinet.utoronto.ca/index.php/Mist#Open-CE) channel. Where a compatible Conda package cannot be found, installing from PyPI (`pip install`) can be attempted. Pip will attempt to compile the package’s source code if no compatible precompiled wheel is available, therefore a compiler module (such as `gcc/.core`) should be loaded in advance. Some packages require tweaking of the source code or build procedure to successfully compile on Mist, please contact [support](https://docs.scinet.utoronto.ca/index.php/Mist#Support) if you need assistance.

# Data Simulation

In ./R_code_simulate_data/data_simulation.R file, 

Adjust the following parameters for customized usage:

```R
# 1 for random tree with padding tips, 0 for fixed tree tips
random_tree_tips = 0 
# number of simulations per CPU process
per_runs = 25000 
# number of simulations in one loop.
runs_in_one_loop = 2500
# tree tips size
t_s = 100 
# Can generate multple time of different number of genes
# if only generate one type of number of genes, then set those two parameters as the same value
# Start Genes Number
start_number_of_genes = 5
# End Genes Number
end_number_of_genes = 5

```
Run such bash file in Niagara Cluster or local device for generate training data by

```
sbatch ./job_submission_bash_template/data_simulation_R_jobs.sh 

or sh data_simulation.sh
```
As the following format:
```Shell

#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=23:59:00
#SBATCH --job-name data_simulation_job
#SBATCH --output=/scratch/*****/zeshengj/logs/5_genes_100_normal_tips_data_simulation_output_%j.txt
#SBATCH --mail-user=zs549061@dal.ca
#SBATCH --mail-type=ALL

module load gcc/8.3.0
module load intel/2019u4
module load r/4.1.2

source ~/.virtualenvs/R_env/bin/activate

Rscript /scratch/***/R_code_simulate_data/data_simulation.R
```


## Run the CNN image data generation by

```
./data_generation.sh
```

### Template of data_generation.sh

```Shell
#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=23:59:00
#SBATCH --job-name data_simulation_job
#SBATCH --output=/scratch/******/zeshengj/logs/5_genes_100_normal_tips_data_simulation_output_%j.txt
#SBATCH --mail-user=zs*****@dal.ca
#SBATCH --mail-type=ALL

module load gcc/8.3.0
module load intel/2019u4
module load r/4.1.2

source ~/.virtualenvs/R_env/bin/activate

python ./permutated_CNN/scripts/generate_data.py \
    --directory /scratch/******/zeshengj/CNN/data/demo \
    --profile_file_name_pattern profiles.csv \
    --rates_file_name_pattern rates.csv \
    --number_of_profiles_in_one_gene_image_array 32 \
    --generation_type 0 \
    --cut_off_files 0 \
    --total_files_to_convert 32 \
    --gene_image_type 1\
    --number_of_genes 40
```

- `--directory` (type: str): Specifies the directory for the raw data. This parameter is required.

- `--profile_file_name_pattern` (type: str): Specifies the pattern for the profile file names. This parameter is required.

- `--rates_file_name_pattern` (type: str): Specifies the pattern for the rates file names. This parameter is required.

- `--number_of_profiles_in_one_gene_image_array` (type: int, default: 6): Specifies the number of profiles in one gene image array. 

- `--generation_type` (type: int, default: 0): Specifies the generation type. 
   - 0: Generates gene images and rates.
   - 1: Generates rates only.
   - 2: Generates gene images only.

- `--cut_off_files` (type: int, default: 0): Specifies whether to generate a certain number of files. 
   - 1: Generates only a certain number of files.
   - 0: Does not perform this operation.

- `--total_files_to_convert` (type: int, default: 32): Specifies the total number of files to convert.

- `--gene_image_type` (type: int, default: 0): Specifies the gene image type. 
   - 0: Permutated gene images.
   - 1: Duplicates gene images.

- `--number_of_genes` (type: int, default: 0): Specifies the number of genes in the data. This parameter is used for automatically padding the image.


## Training by

```
./train.sh
```

### Template of ***train.sh***

```Shell
python ./permutated_CNN/scripts/train.py \
    --main_dir /scratch/******/zeshengj/CNN/data/demo  \
    --if_use_temperary_ssd 0\
    --temperary_ssd_dr /scratch/******/zeshengj/CNN/data/demo/temp \
    --load_model_checkpoint 0 \
    --model_checkpoint_path /scratch/******/zeshengj/CNN/model_checkpoints/m.pth\
    --epochs 100000 \
    --batch_size 256 \
    --num_outputs 780 \
    --sub_training_batch 3\
    --input_gene_image_size "1, 400, 200" \
    --gene_image_type 1 \
    --log_file_name "100_genes_ResNet50_2e-5" \
    --model_type "ResNet" \
    --ResNet_depth 152 \
    --learning_rate 1e-4 \
```

- `--main_dir`:
The mian directory contains the files from R data simulation
- `--if_use_temperary_ssd`
Sometimes, we use external hard driver or HDD to contain the large data files. But the loading time from those disks are slower than SSD. If we have extra SSD, then we can copy some of those large files in advance to the SSD. And the program will iteratively load the most recent need data files during the training, and delete those files after training that batch of files.

	0 for not use temperary SSD
	1 for using it
- `--temperaray_ssd_dr`
If the previous setting is 1, then define the ssd directory path.
- `--load_model_checkpoint`
0 for not loading model checkpoints and create a new model.
1 for loading a model checkpoint
- `--model_checkpoint_path`
define the model checkpoint path location
- `--epochs 100000 `
Number of epochs during the training
- `--batch_size 256` 
Number of images in one batch in GPU. Depends on your GPU memory size.
- `--num_outputs 780 `
Number of Betas in the output. If we have 5 genes, then 5 choose 2 = 10. And if we have 40 genes, then 40 choose 2 = 780.
- `--sub_training_batch 3`
For the scenario when we don't have enough memory to contain all the data files. Then we sub batch train the data. For example, we have 1.2 million dataset, we will divide them into 14 batches with 80K data in each batch. And for each batch, we train the model with 3 epoches.
- `--input_gene_image_size "1, 400, 200" `
For the genes image size
- `--gene_image_type 1 `
1 for duplicates images
0 for permutated images
- `--log_file_name "100_genes_ResNet50_2e-5" `
Create a folder called such name for saving the logs
- `--model_type "ResNet" `
"ResNet" for creating a ResNet model
"CNN" for creating a normal CNN model
- `--ResNet_depth 152 `
    Define ResNet depth.
    * 18 for ResNet 18
    *  34
    * 50
    * 101
    * 152
- `--learning_rate 1e-4 `
Define the corresponding Learning Rate  


## Running on Mist
### Slurm file 

train_jobs.sh
```Shell
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:15:0
#SBATCH --job-name data_generation_job
#SBATCH --output=/scratch/******/zeshengj/logs/data_generation_output_%j.txt

module load anaconda3
source activate genes_env
sh /scratch/******/zeshengj/CNN/gene_corelation_CNN/train.sh
```

Submit job into the queue.
```Shell
sbtach train_jobs.sh
```

Check how long it will start.
```Shell
squeue --start -u user_name
```

Check job status.
```Shell
squeue -u user_name
```

