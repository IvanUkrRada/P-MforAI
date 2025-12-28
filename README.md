# Project Setup

## Overview

This repository contains the setup instructions, task breakdown, and environment configuration for the coursework project. Follow the steps below to correctly configure the environment and keep it updated throughout development.

---
---

## Tasks Breakdown

- **Can be found in 'CW Description.pdf'**

---
---

## Dataset Used

- https://www.kaggle.com/competitions/cifar-10/data (Task 1)
- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (Task 1)
- https://arxiv.org/abs/1912.12142v1 (Task 2)

---
---

# Conda Environment Setup
## For Task 1:

### Setting Up the Coursework Environment Locally

1. **Navigate to the project directory**
   Ensure you are in the directory where the `environment_task1.yml` file is located.

2. **Create the Conda environment (Assuming you have anaconda navigator installed)**

   ```
   conda env create -f environment_task1.yml
   ```

3. **Activate the environment**

   ```
   conda activate IN3063_Coursework_env_task1
   ```

   Example looks like this in terminal:

   ```
   (base) ➜  P-MforAI git:(main) ✗ conda activate IN3063_Coursework_env_task1
   (IN3063_Coursework_env_task1) ➜  P-MforAI git:(main) ✗
   ```

### No changes on yml file.

1. **Simply Activate the environment and start running the model**

   ```
   conda activate IN3063_Coursework_env_task1_task1
   ```

### Updating the Existing Conda Environment

1. **Pull the latest changes**
   Ensure your local repository is up to date with GitHub:

   ```
   git pull
   ```

2. **Activate the existing environment**

   ```
   conda activate IN3063_Coursework_env_task1
   ```

3. **Update the environment**
   This compares your active environment with `environment_task1.yml`, installing missing packages and removing outdated ones ( -- prune is there to delete packages prev installed but not currently in the yml file ):

   ```
   conda env update --file environment_task1.yml --prune
   ```

## Task 1: Running Testing.ipynb or any .ipynb file
 

### Steps to Execute


1. **Open `Testing.ipynb`** in VS Code or Jupyter Notebook


2. **Select the kernel**
  - Click **Select Kernel** (top-right corner)
  - Choose **Python Environments**
  - Select **IN3063_Coursework_env_task1**


3. **Run the notebook**
  - Execute cells using `Shift + Enter`
  - **Note**: On first run, you may be prompted to install `ipykernel`. Click **Install** when prompted.


4. **Verify setup**
  - All cells should execute without errors
  - Check that packages like `numpy`, `matplotlib`, and custom modules load correctly


---
---

## Task-2

 - For Task - 2 Anaconda Navigator was used to run VS code
 # CUDA 13.0
 - pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

---

## Environment for Task 2

1. **Navigate to the project directory**
   
   Ensure you are in the directory where the `task2.yml` file is located.

3. **Create the Conda environment**

   ```
    conda env create -f task2.yml
   ```

3. **Activate the environment**

   ```
   conda activate IN3063_Coursework_env_task1
   ```

    Example looks like this in terminal:

   ```
   (base) ➜  P-MforAI git:(main) ✗ conda activate IN3063_Coursework_env_task1
   (IN3063_Coursework_env_task1) ➜  P-MforAI git:(main) ✗
   ```
   
### No changes on yml file.

1. **Activate the environment and start running the model**

   ```
   conda activate IN3063_Coursework_env_task2
   ```

### Updating the Existing Conda Environment

1. **Pull the latest changes**
   Ensure your local repository is up to date with GitHub:

   ```
   git pull
   ```

2. **Activate the existing environment**

   ```
   conda activate IN3063_Coursework_env_task2
   ```

3. **Update the environment**
   This compares your active environment with `task2.yml`, installing missing packages and removing outdated ones ( -- prune is there to delete packages prev installed but not currently in the yml file ):

   ```
   conda env update --file task2.yml --prune
   ```

---
   
## Collaborators

- **Ivan Radavskyi** 
- **Sujit Bhatta**
- **Wael Khafagi**
- **Yaseen Mneimneih**
- **MD Iftier Roshid**

---
