# Project Setup

## Overview

This repository contains the setup instructions, task breakdown, and environment configuration for the coursework project. Follow the steps below to correctly configure the environment and keep it updated throughout development.

---

## Tasks Breakdown

- **Can be found in 'CW Description.pdf'**

---

## Dataset Used

- https://www.kaggle.com/competitions/cifar-10/data (Task 1)
- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (Task 1)
- https://arxiv.org/abs/1912.12142v1 (Task 2)

---

## Conda Environment for Task 1

### Setting Up the Coursework Environment Locally

1. **Navigate to the project directory**
   Ensure you are in the directory where the `environment.yml` file is located.

2. **Create the Conda environment (Assuming you have anaconda navigator installed)**

   ```
   conda env create -f environment.yml
   ```

3. **Activate the environment**

   ```
   conda activate IN3063_Coursework_env
   ```

   Example looks like this in terminal:

   ```
   (base) ➜  P-MforAI git:(main) ✗ conda activate IN3063_Coursework_env
   (IN3063_Coursework_env) ➜  P-MforAI git:(main) ✗
   ```

### No changes on yml file.

1. **Simply Activate the environment and start running the model**

   ```
   conda activate IN3063_Coursework_env
   ```

### Updating the Existing Conda Environment

1. **Pull the latest changes**
   Ensure your local repository is up to date with GitHub:

   ```
   git pull
   ```

2. **Activate the existing environment**

   ```
   conda activate IN3063_Coursework_env
   ```

3. **Update the environment**
   This compares your active environment with `environment.yml`, installing missing packages and removing outdated ones ( -- prune is there to delete packages prev installed but not currently in the yml file ):

   ```
   conda env update --file environment.yml --prune
   ```

---

## Task-2

 - For Task - 2 Anaconda Navigator was used to run VS code
 # CUDA 13.0
 - pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

---

## Environment for Task 2

1. **Navigate to the project directory**
   
   Ensure you are in the directory where the `cnntsk2.yml` file is located.

3. **Create the Conda environment

```
 conda env create -f cnntsk2.yml
```

3. **Activate the environment**

   ```
   conda activate IN3063_Coursework_env
   ```

    Example looks like this in terminal:

   ```
   (base) ➜  P-MforAI git:(main) ✗ conda activate IN3063_Coursework_env
   (IN3063_Coursework_env) ➜  P-MforAI git:(main) ✗
   ```
   
### No changes on yml file.

1. **Activate the environment and start running the model**

   ```
   conda activate IN3063_Coursework_env
   ```

---
   
## Collaborators

- **Ivan Radavskyi** 
- **Sujit Bhatta**
- **Wael Khafagi**
- **Yaseen Mneimneih**
- **R**

---
