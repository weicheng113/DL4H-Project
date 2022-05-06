# Project for Deep Learning for Healthcare

In this project we aim to replicate the paper ('Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit')[https://arxiv.org/pdf/2007.09483v4.pdf] by Emma Rocheteau, Pietro Li `o and
Stephanie Hyland. The original repository for the paper can be found (here)[https://github.com/EmmaRocheteau/TPC-LoS-prediction]. This repository contains some modifications in order to carry out experiments.

### Setup New Conda Environment
```shell
# to create conda environment.
conda env create -f environment.yml

# to remove conda environment.
conda remove --name dl4h-project --all

# to update conda environment when some new libraries are added.
conda env update -f environment.yml --prune
```

### Download eICU Data

Download eICU Data from pyhsionet - https://physionet.org/content/eicu-crd/2.0/. 

### Setup eICU Database Locally
```shell
# 1. Install Postgres http://www.postgresql.org/download/

# 2. Create database
# 2.1 start SQL Shell(psql)
# 2.2 create tables
\i [path to eicu-code]/build-db/postgres/postgres_create_tables.sql
# 2.3 navigate to data directory
\cd [path to eicu-collaborative-research-database-2.0]
# 2.4 load data
\i [path to eicu-code]/build-db/postgres/postgres_load_data_gz.sql
# 2.5 add indices
\i [path to eicu-code]/build-db/postgres/postgres_add_indexes.sql
# 2.6 validate
\i [path to eicu-code]/build-db/postgres/postgres_checks.sql
# 2.7 create views
\cd [path to DL4H-Project/]
\i eICU_preprocessing/create_all_tables.sql
```

### Run Preprocessing
```shell
# 1. activate dl4h-project Conda environment.
conda activate dl4h-project

# 2. run the pre-processing scripts
python -m eICU_preprocessing.run_all_preprocessing

# 3. modify paths.json 
{"eICU_path": "[path to eICU_data folder produced by preprocessing step]"}

# 4. create lmdb databases to supports random data item access.
python csv2lmdb.py
```



