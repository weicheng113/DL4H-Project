# DL4H-Project
Project for Deep Learning for Healthcare

### Setup Environment
```shell
# install Microsoft C++ Build Tools
# Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select 'Desktop development with C++'

# to create conda environment.
conda env create -f environment.yml

# to remove conda environment.
conda remove --name dl4h-project --all

# to update conda environment when some new libraries are added.
conda env update -f environment.yml --prune
```
### Setup eICU Database Locally
```shell
# 1. download files https://physionet.org/content/eicu-crd/2.0/.

# 2. install Postgres http://www.postgresql.org/download/

# 3. clone https://github.com/MIT-LCP/eicu-code

# 4. start SQL Shell(psql)
# 4.1 create tables
\i [path to eicu-code]/build-db/postgres/postgres_create_tables.sql
# 4.2 navigate to data directory
\cd [path to eicu-collaborative-research-database-2.0]
# 4.3 load data
\i [path to eicu-code]/build-db/postgres/postgres_load_data_gz.sql
# 4.4 add indices
\i [path to eicu-code]/build-db/postgres/postgres_add_indexes.sql
# 4.5 validate
\i [path to eicu-code]/build-db/postgres/postgres_checks.sql
# 4.6 create views
\cd [path to DL4H-Project/TPC-LoS-prediction/]
\i eICU_preprocessing/create_all_tables.sql

# 5. run the pre-processing scripts
python -m eICU_preprocessing.run_all_preprocessing
```

### Setup eICU Database Locally
```shell
# 1. download files https://physionet.org/content/mimiciv/0.4/.

# 2. install Postgres http://www.postgresql.org/download/

# 3. clone https://github.com/EmmaRocheteau/MIMIC-IV-Postgres

# 4. start SQL Shell(psql)
# 4.1 create tables
\i D:/git/MIMIC-IV-Postgres/create_tables.sql
# 4.2 load data
\i D:/git/MIMIC-IV-Postgres/load_data_gz.sql
```