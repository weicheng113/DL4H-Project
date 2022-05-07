# Project for Deep Learning for Healthcare

In this project we aim to replicate the paper ['Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'](https://arxiv.org/pdf/2007.09483v4.pdf) by Emma Rocheteau, Pietro Li `o and
Stephanie Hyland. The original repository for the paper can be found [here](https://github.com/EmmaRocheteau/TPC-LoS-prediction). This repository contains some modifications in order to carry out experiments.

### Setup New Conda Environment

Use the first command to create new independent environment for the project. Or use the other two commands to remove or update the Conda environment.

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

Follow the steps below to set up the database locally.

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

### Pre-processing

Follow the steps below to pre-process the data and prepare the data for training.

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

### Training

To train the models, run one of following commands.

```shell
# a. To train various TPC models. Model choices --model: [tpc, tpc-multitask, tpc-mse, pointwise-only, temp-only, tpc-no-skip, "tpc-mask-skip"]
python train_tpc.py --model tpc

# b. To train channel-wise LSTM model.
python train_lstm.py

# c. To train transformer model.
python train_transformer.py
```

### Evaluation

To test the models, run one of following commands.

```shell
# a. To test various TPC models. Model choices --model: [tpc, tpc-multitask, tpc-mse, pointwise-only, temp-only, tpc-no-skip, "tpc-mask-skip"]
python test_tpc.py --model tpc

# b. To test channel-wise LSTM model.
python test_lstm.py

# c. To test transformer model.
python test_transformer.py
```

### Results

Below is the result comparison for TPC model between the original paper and our replication experiment on eICU test data.

| **Model**           | **MAD**   | **MSE**   | **MAPE** | **MSLE**  | **R^2**   | **KAPPA** |
|---------------------|-----------|-----------|----------|-----------|-----------|-----------|
| TPC(original paper) | 1.78±0.02 |  21.7±0.5 | 63.5±4.3 | 0.70±0.03 | 0.27±0.02 | 0.58±0.01 |
| TPC(replication)    | 1.658     | 19.539    | 49.697   | 0.458     | 0.443     | 0.710     |

Below is the result comparison for TPC model vs other baseline models.

| **Model**         | **MAD** | **MSE** | **MAPE** | **MSLE** | **R^2** | **KAPPA** |
|-------------------|---------|---------|----------|----------|---------|-----------|
| TPC               | 1.658   | 19.539  | 49.697   | 0.458    | 0.443   | 0.710     |
| Channel-wise LSTM | 2.631   | 31.825  | 115.745  | 1.424    | 0.096   | 0.324     |
| Transformer       | 2.592   | 30.617  | 126.388  | 1.437    | 0.127   | 0.340     |

Below is the result comparison for TPC model vs other TPC variant models.

| **Model**          | **MAD** | **MSE** | **MAPE** | **MSLE** | **R^2** | **KAPPA** |
|--------------------|---------|---------|----------|----------|---------|-----------|
| TPC<img width=100/>| 1.658   | 19.539  | 49.697   | 0.458    | 0.443   | 0.710     |
| TPC(multitask)     | 1.241   | 16.058  | 27.520   | 0.214    | 0.542   | 0.821     |
| TPC(no skip)       | 1.918   | 22.929  | 64.399   | 0.716    | 0.346   | 0.627     |
| Temp. only         | 1.798   | 21.437  | 57.617   | 0.615    | 0.389   | 0.665     |
| Point. only        | 2.665   | 31.458  | 110.025  | 1.517    | 0.106   | 0.336     |
| TPC(MSE)           | 2.107   | 20.965  | 144.790  | 1.565    | 0.402   | 0.639     |

Below is the result comparison for TPC model vs an additional TPC with mask field included in skip connection.

| **Model**          | **MAD** | **MSE** | **MAPE** | **MSLE** | **R^2** | **KAPPA** |
|--------------------|---------|---------|----------|----------|---------|-----------|
| TPC<img width=100/>| 1.658   | 19.539  | 49.697   | 0.458    | 0.443   | 0.710     |
| TPC(mask skip)     | 1.267   | 15.385  | 30.231   | 0.222    | 0.561   | 0.817     |

### Citation

```bibtex
@inproceedings{rocheteau2021,
author = {Rocheteau, Emma and Li\`{o}, Pietro and Hyland, Stephanie},
title = {Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit},
year = {2021},
isbn = {9781450383592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450439.3451860},
doi = {10.1145/3450439.3451860},
booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
pages = {58–68},
numpages = {11},
keywords = {intensive care unit, length of stay, temporal convolution, mortality, patient outcome prediction},
location = {Virtual Event, USA},
series = {CHIL '21}
}
```
