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

### Setup MIMIC-IV Database Locally
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



### Setup MIMIC-IV Database Locally
```shell


# 1. download files https://physionet.org/content/mimiciv/0.4/.

# 2. install Postgres http://www.postgresql.org/download/

# 3. clone https://github.com/EmmaRocheteau/MIMIC-IV-Postgres

# 4. Open command prompt 

# 5. Write the following commands:

	psql -U postgres 

	CREATE DATABASE mimic4;      #create new database
	\c mimic4;                   #connect to the new database
	CREATE SCHEMA mimiciv;       #create new schema
	\q   

# 6. Create a set of empty tables with create_tables.sql 
#   Load "create_tables.sql" in pgadmin as a query and run

# 7. Load .csv files into the empty tables with load_data.sql

# 8. Clone the https://github.com/EmmaRocheteau/TPC-LoS-prediction repository

# 9. Open cmd and navigate to the repository and run the following commands:

	psql "dbname=mimic4 user=postgres options=--search_path=mimiciv"
	CREATE EXTENSION tablefunc schema mimiciv;  #(required for crosstab function)
	set search_path = mimiciv;

# 10. Run the create_tables sql file
	\i MIMIC_preprocessing/create_all_tables.sql;

# 11. Run the python pre-processing file
	python -m MIMIC_preprocessing.run_all_preprocessing

# 12. 2,310,510 rows in preprocessed_timeseries.csv (Output : 5GB file)
# Stuck at processing 24,000 patients. No error

```