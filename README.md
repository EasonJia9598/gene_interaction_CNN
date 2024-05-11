# Data Generation

# Training

# Mist cluster Instructions
## Zesheng Jia
## For install packages

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
cd /scratch/h/honggu/zeshengj/CNN/
pip install -e .
```

For Mist Cluster, we need to manually install any packages in requirements from conda.
By the following reasons.

> **Important note:** the majority of computer systems as of 2021 (laptops, desktops, and HPC) use the 64 bit x86 instruction set architecture (ISA) in their microprocessors produced by Intel and AMD. This ISA is incompatible with Mist, whose hardware uses the 64 bit PPC ISA (set to little endian mode). The practical meaning is that x86-compiled binaries (executables and libraries) cannot be installed on Mist. For this reason, the Niagara and Alliance (formerly Compute Canada) software stacks (modules) cannot be made available on Mist, and using closed-source software is only possible when the vendor provides a compatible version of their application. **Python applications** almost always rely on bindings to libraries originally written in C or C++, some of them are not available on PyPI or various Conda channels as precompiled binaries compatible with Mist. ***The recommended way to use Python on Mist is to create a [Conda](https://docs.scinet.utoronto.ca/index.php/Mist#Anaconda_.28Python.29) environment and install packages from the anaconda (default) channel, where most popular packages have a linux-ppc64le (Mist-compatible) version available.*** Some popular machine learning packages should be installed from the internal [Open-CE](https://docs.scinet.utoronto.ca/index.php/Mist#Open-CE) channel. Where a compatible Conda package cannot be found, installing from PyPI (`pip install`) can be attempted. Pip will attempt to compile the package’s source code if no compatible precompiled wheel is available, therefore a compiler module (such as `gcc/.core`) should be loaded in advance. Some packages require tweaking of the source code or build procedure to successfully compile on Mist, please contact [support](https://docs.scinet.utoronto.ca/index.php/Mist#Support) if you need assistance.

## Run the data generation by

```
./data_generation.sh
```

### Template of data_generation.sh

```Shell
python ./permutated_CNN/scripts/generate_data.py \
    --directory /scratch/h/honggu/zeshengj/CNN/data/demo \
    --profile_file_name_pattern profiles.csv \
    --rates_file_name_pattern rates.csv \
    --number_of_profiles_in_one_gene_image_array 32 \
    --generation_type 0 \
    --cut_off_files 0 \
    --total_files_to_convert 32 \
    --gene_image_type 1\
    --number_of_genes 40
```

```Shell
parser.add_argument('--directory', type=str, help='directory for the raw data')
parser.add_argument('--profile_file_name_pattern', type=str, help='pattern for the profile file names')
parser.add_argument('--rates_file_name_pattern', type=str, help='pattern for the rates file names')
parser.add_argument('--number_of_profiles_in_one_gene_image_array', type=int, default=6, help='number of profiles in one gene image array')
parser.add_argument('--generation_type', type=int, default=0, help='generation type, 0 for generating genes images and rates, 1 for generating rates, 2 for generating gene images')
parser.add_argument('--cut_off_files', type=int, default=0, help='1 for only generating certain number of files, 0 for doing no such operation.')
parser.add_argument('--total_files_to_convert', type=int, default=32, help='total files to convert')
parser.add_argument('--gene_image_type', type=int, default=0, help='gene image type, 0 for permutated gene images, 1 for duplicates gene images')
parser.add_argument('--number_of_genes', type=int, default=0, help='Number of genes in the data. It will automatically padding the image.')
```

## Training by

```
./train.sh
```

### Template of ***train.sh***

```Shell
python ./permutated_CNN/scripts/train.py \
    --main_dir /scratch/h/honggu/zeshengj/CNN/data/demo  \
    --if_use_temperary_ssd 0\
    --temperary_ssd_dr /scratch/h/honggu/zeshengj/CNN/data/demo/temp \
    --load_model_checkpoint 0 \
    --model_checkpoint_path /scratch/h/honggu/ze_shengj/CNN/model_checkpoints/m.pth\
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

##### --main_dir:
The mian directory contains the files from R data simulation
##### --if_use_temperary_ssd
Sometimes, we use external hard driver or HDD to contain the large data files. But the loading time from those disks are slower than SSD. If we have extra SSD, then we can copy some of those large files in advance to the SSD. And the program will iteratively load the most recent need data files during the training, and delete those files after training that batch of files.

	0 for not use temperary SSD
	1 for using it
##### --temperaray_ssd_dr
If the previous setting is 1, then define the ssd directory path.
##### --load_model_checkpoint
0 for not loading model checkpoints and create a new model.
1 for loading a model checkpoint
##### --model_checkpoint_path
define the model checkpoint path location
##### --epochs 100000 
Number of epochs during the training
##### --batch_size 256 
Number of images in one batch in GPU. Depends on your GPU memory size.
##### --num_outputs 780 
Number of Betas in the output. If we have 5 genes, then 5 choose 2 = 10. And if we have 40 genes, then 40 choose 2 = 780.
##### --sub_training_batch 3
For the scenario when we don't have enough memory to contain all the data files. Then we sub batch train the data. For example, we have 1.2 million dataset, we will divide them into 14 batches with 80K data in each batch. And for each batch, we train the model with 3 epoches.
##### --input_gene_image_size "1, 400, 200" 
For the genes image size
##### --gene_image_type 1 
1 for duplicates images
0 for permutated images
##### --log_file_name "100_genes_ResNet50_2e-5" 
Create a folder called such name for saving the logs
##### --model_type "ResNet" 
"ResNet" for creating a ResNet model
"CNN" for creating a normal CNN model
##### --ResNet_depth 152 
Define ResNet depth.
1. 18 for ResNet 18
2. 34
3. 50
4. 101
5. 152
##### --learning_rate 1e-4 
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
#SBATCH --output=/scratch/h/honggu/zeshengj/logs/data_generation_output_%j.txt

module load anaconda3
source activate genes_env
sh /scratch/h/honggu/zeshengj/CNN/gene_corelation_CNN/train.sh
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

