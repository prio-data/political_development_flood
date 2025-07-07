
# Replicating Vesco et al., *"Political Development Predicts Reduced Human Cost of Flooding"*

## Overview

This repository contains the datasets, R scripts, and instructions to replicate the findings presented in the manuscript and supplementary materials. 
The study explores the role of political development in mitigating the human cost of flooding using Bayesian statistical modeling.

### System Requirements and Setup

The analysis was run with the following system specifications: 

- **Operating System**: macOS Sonoma 14.4.1
- **R Version**: 4.3.1
- **RStudio Version**: 2023.12.1+402


## Repository Structure


**Data Folder**

The data/ folder contains:

1. data_final.rds: The main dataset used for analysis.
2. Additional datasets for generating maps in the descriptive analysis and for conducting the counterfactual analysis.

**Code Folder** 

The code/ folder is structured in three main scripts:

1. descriptives.rda:

- Purpose: Produces descriptive tables and figures, including maps, summary statistics, and correlation plots.

2. analysis.rda:

- Purpose: Replicates the analysis and generates results for the manuscript and supplementary materials.

3. rf.R:

- Purpose: Trains the random forest models and computes Shapley values for feature importance, as detailed in the Supplementary Material.

**Results Folder** 

The results/ folder stores the results and it is created by the analysis.Rda script automatically. The results folder is organised in two main subfolders:

1. descriptives: contains all descriptive figures and tables. 
2. panel: contains the results of the main analysis presented in the manuscript as well as for the sensitivity tests. It is organised in different sub-folders according to the model specification. The subfolder "GDP" stores the results for the main specification, the other sub-folders contain the results for the sensitivity tests.


## INSTRUCTIONS TO RUN THE ANALYSIS

## Settings and requirements

Run the requirements.R script to install all required packages, or execute:
`source("code/requirements.R")`

To install some packages (e.g., brms) directly from GitHub, you need a GitHub Personal Access Token (PAT). If installing from GitHub fails, ensure your PAT is correctly set up.
For assistance, refer to the official GitHub documentation: Creating a Personal Access Token. Brief instructions are reported here:

- Log in to your GitHub account.
- Navigate to Personal Access Tokens.
- Click Generate new token (classic).
- Provide a descriptive name and select the following scopes:
     -- repo (Full control of private repositories)
     -- read:packages

- Click Generate token and copy the token. 
- Save it securely; you won’t be able to see it again.

- Install the gitcreds package by running:

`install.packages("gitcreds")`

`library(gitcreds)`

`gitcreds_set()`

- Paste the token when prompted.


## Running the analysis

**Run the descriptive analysis:**

Execute descriptives.rda to generate the descriptive statistics and figures.

- How to Run: Open the script in RStudio, set the working directory to the repository root, and execute the script. Outputs are saved in the results/ folder.

- Outputs include maps, correlation plots, and summary statistics.

**Run the main analysis presented in the manuscript:**

Execute analysis.rda to replicate the main analysis and generate results.

- How to Run: Open the script in RStudio, set the working directory to the repository root, and execute the script. Outputs are saved in the results/ folder.

- Additional instructions:

By default, the script runs with `GDP = TRUE` to produce the main results reported in the manuscript. The default setting does not re-train the model but loads the pre-trained models stored in the results folder. 
We recommend this setting for faster compilation.

To explore alternative model specifications as outlined in the Supplementary Information, adjust the parameters at the beginning of the script. For example:
NOCHI <- TRUE  # Exclude China
GDP <- FALSE  # Disable the main specification

Ensure all other parameters are set to `FALSE` unless required.


**Run the random forest analysis:**

Execute rf.R to train random forest models and compute Shapley values.

- How to Run: Open the script in RStudio, set the working directory to the repository root, and execute the script. Outputs are saved in the results/ folder.

- Outputs: Results are saved in the results/ folder, organized by script and specification.

## Approximate runtime

- Model Training: Bayesian model training can take 3–24 hours, depending on system resources.
- Descriptive Analysis: ~15 minutes.
- Random Forest Analysis: ~2 hours.

## Reproducibility

Due to the parallel processing generating some uncertainty when training the bayesian mdoels, we expect slight variations in results if the models are re-trained, even with fixed seeds (set.seed()). 
These minor differences do not affect the validity of the conclusions.

## Support

If you encounter issues or have questions, contact P. Vesco at paoves@prio.org

