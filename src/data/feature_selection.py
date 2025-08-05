### Dependencies ###
import pandas as pd
import numpy as np
import os

import importlib

from scipy.stats import shapiro, kstest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV,ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, classification_report, f1_score, matthews_corrcoef, mean_squared_error,r2_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import kstest_normal
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2
from timeit import default_timer as timer

### Paths ###
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
output_dir = os.path.join(project_root, 'data', 'data_output')
os.makedirs(output_dir, exist_ok=True)


### Functions ###
def binary_labeling(df: pd.DataFrame, classified_by: str, true_class: list ) -> pd.DataFrame:
    """
    Adds a binary classifier column to the DataFrame based on the specified true classes.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    classified_by (str): The column name containing the classification labels.
    true_class (str): The class label to be considered as '1' in the binary
    
    """
    if 'Classifier' in df.columns:
        df = df.drop(columns=['Classifier'])
        
    df.insert(
                loc=2,  # position 2
                column='Classifier',
                value=np.where(df[classified_by].isin(true_class), 1, 0)
            )
    
    print (f"\nNumber of samples per class:\n{df['Classifier'].value_counts()}\n")


    return df


def get_high_confidence_proteins(peptide_binary_df:pd.DataFrame, true_class:list, classified_by:str, threshold=0.7) -> list[str] :
    """
    Returns a list of protein names with identification in 70% of samples with at leat 2 peptides.
     
    Args:
        peptide_binary_df (pd.DataFrame): DataFrame with binary peptide data.
        true_class (list): List of classes to filter the DataFrame.
        classified_by (str): Column name used for classification.
        threshold (float): Minimum percentage of samples in which a protein must be identified to be included
    
    Returns:
        list[str]: List of protein names that meet the criteria.
    """
    class_peptides = peptide_binary_df[peptide_binary_df[classified_by].isin(true_class)]
    class_peptides_stats = class_peptides.describe()
    proteins_by_peptides = list(
        class_peptides_stats.T[class_peptides_stats.T['mean'] >= threshold].index
    )
    print (f" {len(proteins_by_peptides)} proteins identified in {threshold*100}% of {true_class} samples")
    
    return proteins_by_peptides


def hparameter_grid_search(df: pd.DataFrame, n_splits: int, l1_ratio_list: list, C_list: list, classified_by: str) -> tuple:
    """
    Perform grid search for logistic regression using the provided dataframe, number of splits for cross-validation, 
    l1_ratio and C values.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data. Provide main training df for this specific type.
        n_splits (int): Number of splits for Stratified K-Fold cross-validation. According to sample number
        l1_ratio_list (list): List of l1_ratio values for elasticnet
        C_list (list): List of regularization strength values for Logistic Regression (C)

    Returns:
    grid_search.cv_results_ (dict): CV results from grid search
    grid_search.best_params_ (dict): Best parameters from grid search
    grid_search.best_score_ (float): Best score from grid search
    grid_search (GridSearchCV object): The fitted grid search object for further analysis
    """
    start = timer()

    # Selecting Data
    y_train = df['Classifier']  # True values (dependent variable)
    X_train = df.drop(columns=['Sample name', 'Classifier', classified_by], axis=1)  # Protein (independent variables) (Keep only quantitative data)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'penalty': ['elasticnet'],
        'l1_ratio': l1_ratio_list,  # l1_ratio input by user
        'solver': ['saga'],
        'max_iter': [10000],
        'C': C_list  # C input by user
    }

    scorer = make_scorer(matthews_corrcoef)

    # Create a logistic regression model
    logistic_regression = LogisticRegression(warm_start=True)

    # Define Stratified K-Fold Cross-Validation
    stratified_kfold = StratifiedKFold(n_splits=n_splits, 
                                       shuffle=True, 
                                       random_state=0)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(
        logistic_regression, param_grid, 
        cv=stratified_kfold,
        scoring=scorer, 
        n_jobs=8, 
        return_train_score=True
    )

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    end = timer()
    print(f"Grid search completed in {end - start:.2f} seconds")
    
    # Print the results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # Return the results
    return grid_search.cv_results_, grid_search.best_params_, grid_search.best_score_, grid_search





