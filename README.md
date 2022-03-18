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
