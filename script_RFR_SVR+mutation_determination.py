#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# In[2]:


# Read the dataset into a DataFrame
df=pd.read_csv('PI dataset.txt' , sep= '\t', index_col = 'SeqID')


# In[3]:


# Extract features
X = df.iloc[:,0:1159]

# Define column names of target variables
target_variables = ['FPV','ATV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV','DRV']


# In[4]:


# Define the parameter grid for GridSearchCV for Random Forest Regression

param_grid_rf = {'n_estimators' : [50, 70, 90, 100, 128],
                 'max_features' : [ 50, 60, 70, 75, 80, 85],
                 'bootstrap' : [True],
                 'oob_score' : [True]
                }


# In[5]:


#create a list with feature significance
feature_significance_list=[]

# Function to remove unimportant features based on Random Forest feature importances

def remove_unimportant_features(X, y):
    # Create the Random Forest Regressor and the GridSearchCV object
    rf_d = RandomForestRegressor()
    grid_search = GridSearchCV(rf_d, param_grid_rf, cv=5, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)

    # Get the best estimator and its feature importances
    best_estimator = grid_search.best_estimator_
    feature_importances = best_estimator.feature_importances_
    
    #add significance value in the list
    feature_significance_list.append(feature_importances)
    
    # Determine the unimportant features based on the threshold
    threshold = 0.0001  # Threshold below which features are considered unimportant
    unimportant_features = X.columns[feature_importances < threshold]
    
    # Drop the unimportant features from the input data
    X.drop(unimportant_features, axis = 1, inplace = True)

    return X


# In[6]:


#Random Forest Regression

# Initialize lists to store evaluation metrics
r2_rf, rmse_rf, mae_rf = [], [], []

# Loop through each target variable
for target_var in target_variables:
    
    # Defining the target variable y
    y = df[target_var]

    # Remove unimportant features from input data
    X_new = remove_unimportant_features(X.copy(), y)

    # Train Random Forest model with GridSearchCV
    rf = RandomForestRegressor()
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv = 5, n_jobs = -1)
    grid_search_rf.fit(X_new, y)

    # Cross-validation for evaluation metrics
    cv_scores_r2_rf = cross_val_score(grid_search_rf.best_estimator_, X_new, y, cv = 5, scoring = 'r2')
    mean_cv_score_r2_rf = np.mean(cv_scores_r2_rf)
    r2_rf.append(mean_cv_score_r2_rf)

    cv_scores_rmse_rf = cross_val_score(grid_search_rf.best_estimator_, X_new, y, cv = 5, scoring = 'neg_mean_squared_error')
    mean_cv_score_rmse_rf = np.mean(np.sqrt(-cv_scores_rmse_rf))
    rmse_rf.append(mean_cv_score_rmse_rf)

# Create DataFrame to store results of RFR
rf_df_new = pd.DataFrame({'Drug': target_variables, 'RFR, R2': r2_rf, 'RFR, RMSE': rmse_rf})
rf_df_new


# In[7]:


#Support Vector Regression

# Initializing empty lists for R2 and RMSE scores
r2_svr, rmse_svr = [], []

# Loop for each target variable
for target_var in target_variables:
    
    # Defining the target variable y
    y = df[target_var]
    
    # Creating and configuring the SVR model
    svr = SVR(C = 100, gamma ='scale', kernel = 'rbf')
    
    # Evaluating the model using cross-validation for R2
    cv_scores_r2_svr = cross_val_score(svr, X, y, cv = 5, scoring = 'r2')    
    mean_cv_score_r2_svr = cv_scores_r2_svr.mean()
    r2_svr.append(mean_cv_score_r2_svr)
    
    # Evaluating the model using cross-validation for RMSE
    cv_scores_rmse_svr = cross_val_score(svr, X, y, cv = 5, scoring = 'neg_mean_squared_error')
    mean_cv_score_rmse_svr = np.mean(np.sqrt(-cv_scores_rmse_svr))
    rmse_svr.append(mean_cv_score_rmse_svr)

# Create DataFrame to store results of SVR
svr_df_new = pd.DataFrame({'Drug': target_variables, 'SVR, R2': r2_svr, 'SVR, RMSE': rmse_svr})
svr_df_new
    
   


# In[8]:


#Create a DataFrame with feature significance for all peptides and 8 drugs
significance_df=pd.DataFrame({'Peptide': df.columns[:-8]}) 
for i, col in enumerate(target_variables):
    significance_df[col] = feature_significance_list[i]

significance_df=significance_df.set_index('Peptide')
significance_df


# In[9]:


#find top 20 peptide with the highest significance
top20_peptides=[]
for k in significance_df.columns:
    top20_peptides.append(list(significance_df.nlargest(20, k).index))


# In[10]:


#reference HIV protease sequence without mutations to determine peptides positions
protease_reference = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

#find positions of peptides in the protease sequence:
def find_peptides_in_text(text, peptides):
    found_peptides = []
    
    for peptide in peptides:
        best_match_count = 0
        best_start_idx = 0
        
        for i in range(len(text) - len(peptide) + 1):
            match_count = sum(text[i+j] == peptide[j] for j in range(len(peptide)))
            if match_count > best_match_count:
                best_match_count = match_count
                best_start_idx = i + 1
        
        if best_match_count >= len(peptide) - 2:
            found_peptides.append((peptide, (best_start_idx, best_start_idx+len(peptide)-1)))
    
    return found_peptides

for i in range(8):
    peptides = top20_peptides[i]  

    found_peptides = find_peptides_in_text(protease_reference, peptides)

    data = {f"Peptides_{target_variables[i]}": [peptide[0] for peptide in found_peptides],
            f"Coordinates_{target_variables[i]}": [peptide[1] for peptide in found_peptides]}

    df_coordinates = pd.DataFrame(data)
    df_coordinates = df_coordinates.sort_values(by=f"Coordinates_{target_variables[i]}")
    print(df_coordinates)

