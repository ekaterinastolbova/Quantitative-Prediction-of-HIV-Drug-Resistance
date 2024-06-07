# Supplementary-Materials---Quantitative-Prediction-of-Human-Immunodeficiency-Virus-Drug-Resistance
--------
This repository contains Python code for evaluating machine learning models on a dataset with pentapeptides as descriptors and ln(FR) as a target. The code includes implementations of Random Forest Regression (RFR) and Support Vector Regression (SVR) models.

## Overview
The code is designed to evaluate the performance of these models on a set of target variables, including FPV, ATV, IDV, LPV, NFV, SQV, TPV, and DRV. The models are trained and evaluated using cross-validation, and the results are stored in DataFrames for easy analysis.

## Files
- `PI dataset.txt`: The dataset used for training and evaluation.
- `PI sequence dataset.txt` The filtered dataset that contains initial protein sequences of HIV protease and FR. Sequences of variants containing several inaccurately defined amino acid residues at positions of significant drug resistance mutations were excluded
- `script RFR SVR.py`: The  Python script that contains the code for training and evaluating the models.

## Usage
1. Install necessary libraries: Ensure that you have the necessary libraries installed, including **pandas**, **numpy**, **scikit-learn**, and **matplotlib**.
2. Run the script: Run the `script RFR SVR.py` script to execute the code and generate the evaluation results.

## Results
The results of the evaluation are stored in two DataFrames: `rf_df_new` for Random Forest Regression and `svr_df_new` for Support Vector Regression. These DataFrames contain the R2 and RMSE scores for each target variable.
