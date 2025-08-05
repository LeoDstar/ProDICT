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
from joblib import Parallel, delayed
from tqdm import tqdm  

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


def elnet_cross_val (df:pd.DataFrame, classified_by:str, l1_ratio:float, C:float, n_splits:int, random_state=1) -> pd.DataFrame:
    """
    This function return the coefficients of the independent variables defined by ML Logistic Regression. Uses 5k Cross Validation. 
    df:input the imputated and concatenated dataframe, generated with previous functions.
    random_state: random state of CV

    """

    log_reg_coeff_list = []
    results = []

    
    y_train = df['Classifier'] 
    X_train = df.drop(columns=['Sample name', 'Classifier', classified_by], axis=1) 

    #Defining model parameters
    log_reg = LogisticRegression(penalty='elasticnet',  
                                 solver='saga', 
                                 l1_ratio=l1_ratio ,
                                 max_iter=10000, 
                                 C = C, 
                                 class_weight= 'balanced',
                                 warm_start=False,
                                )
    #Saving the id used for each fold:
        
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(X_train, y_train):
        x_train_fold, x_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        log_reg.fit(x_train_fold, y_train_fold)
        
    #Logaritmic regression model coeffiecient:
        log_reg_coeff_list = log_reg.coef_.tolist()
    
    #Logaritmic regression model coeffiecient INTERCEPT:
        log_reg_coeff_list[0].append (log_reg.intercept_[0])
        
    #Model predictions for score in folds
        y_predict_fold = log_reg.predict(x_test_fold)
                
    #Intercept        
        results.append(log_reg_coeff_list[0])
        
    #F1_score for class 1 - entity prediction
        f1_score_class_1 = f1_score(y_test_fold, y_predict_fold, pos_label=1)
        log_reg_coeff_list[0].append(f1_score_class_1)
    
    #F1_score for class 0 - entity prediction
        f1_score_class_0 = f1_score(y_test_fold, y_predict_fold, pos_label=0)
        log_reg_coeff_list[0].append(f1_score_class_0)
    
    #F1_score for class 1 - entity prediction
        f1_score_class_weighted = f1_score(y_test_fold, y_predict_fold, average= 'weighted')
        log_reg_coeff_list[0].append(f1_score_class_weighted)
    
    #MCC Score
        MCC_score = matthews_corrcoef(y_test_fold, y_predict_fold)
        log_reg_coeff_list[0].append(MCC_score)


    col_names = X_train.columns.tolist()
    col_names.extend(['Intercept','F1_1', 'F1_0','F1_weighted','MCC_score' ])
    
    
    coeff_score_df = pd.DataFrame(results)
    coeff_score_df.columns = col_names
    
    return(coeff_score_df)


def elnet_wrapper (df:pd.DataFrame, 
                             classified_by:str,
                             tumor_type_name:str,
                             l1_ratio:float, 
                             C:float,
                             n_splits=4, 
                             n_repeats=1, 
                             n_jobs=8,
                             export=True) -> pd.DataFrame:
    
    """ 
    Wrapper funtion of elnet_cross_val. Executes "n_repeats" times a cross validated logistic regression, storing the coefficients and scores for each "n_repeats" fit of the data.
    An adaptation of bootstrapping
    
    Args:
        df: Input dataframe with proteins intensities, imputated and with no NaN values
        n_repeats: input the number of repetitions. n_repeats = 1 : 1 experiment with 5k Cross validation
        tumor_type_name: Name of the tumor type for which the coefficients are being calculated.
        l1_ratio: L1 ratio for the ElasticNet regularization.
        C: Inverse of regularization strength; smaller values specify stronger regularization.
        n_splits: Number of splits for cross-validation.
        n_jobs: Number of jobs to run in parallel. Default is 8.
        export: Boolean to indicate whether to export the results to an Excel file.
        tumor_type_name: Name of the tumor type for which the coefficients are being calculated.
    Returns:
        df_concatenated: DataFrame containing the concatenated results of the coefficients and scores from multiple runs of the logistic regression.
    """

    if not isinstance(n_repeats, int) or n_repeats <= 0:
        raise ValueError("The number of repetitions 'n_repeats' must be a positive integer.")
    if df.empty:
        raise ValueError("Input dataframe 'df' is empty. Please provide a valid dataframe.")

    repetitions = range(n_repeats)#number of times the ML algorithm will run. triesx5 = #coefficients


    results = Parallel(n_jobs=n_jobs)(
        delayed(elnet_cross_val)(df, 
                                 classified_by=classified_by, 
                                 l1_ratio=l1_ratio, 
                                 C=C, 
                                 random_state=iteration, 
                                 n_splits=n_splits) for iteration in tqdm(repetitions, desc="Running Logistic Regression", unit="iteration")
    )

    # Concatenate the results into one DataFrame
    df_concatenated = pd.concat(results, ignore_index=True)

    #Export
    if export:
        df_concatenated.to_excel(os.path.join(output_dir, f'{tumor_type_name}_coefficients.xlsx'), index=False)
        print(f'DataFrame exported to: {os.path.join(output_dir,  f'{tumor_type_name}_coefficients.xlsx')}')

    return (df_concatenated)



