# Project Setup

## Overview

This repository contains the setup instructions, task breakdown, and environment configuration for the coursework project. Follow the steps below to correctly configure the environment and keep it updated throughout development.

---

## Tasks Breakdown

- **Can be found in 'CW Description.pdf'**

---

## Dataset Used

Provide details about the dataset here (source, format, size, preprocessing steps, etc.).

---

## Conda Environment

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

## How to edit the report

1. Download 'Report VX.docx' (X = the version of the report being edited).
2. Open in Word (preferred).
3. Click 'View' -> 'Outline'.
4. Edit in Outline as it is the easiest way. Use 'Page Break' to start a new page, each new topic should start on a new page.
5. Once done, push to origin.

---

## How to set-up GitHub Desktop

Follow these tutorials, I recommend the second link.
https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/setting-up-github-desktop
https://www.youtube.com/watch?v=8Dd7KRpKeaE&t=88s

---

## Collaborators

- **Sujit Bhatta**
- **Wael Khafagi**

---
