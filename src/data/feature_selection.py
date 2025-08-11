### Dependencies ###
import pandas as pd
import numpy as np
import os

import importlib


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import  make_scorer,  f1_score, matthews_corrcoef
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2
from timeit import default_timer as timer
from joblib import Parallel, delayed
from tqdm import tqdm  
import warnings
from sklearn.exceptions import ConvergenceWarning



### Paths ###
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
output_dir = os.path.join(project_root,'tumor_type_prediction', 'data', 'data_output', 'feature_selection')
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
    logistic_regression = LogisticRegression(warm_start=True,
                                             random_state=93)

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
                                 random_state=93
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
        df_concatenated.to_excel(os.path.join(output_dir, f'{tumor_type_name}_coefficients.xlsx'), engine='xlsxwriter', index=False)
        print(f'DataFrame exported to: {os.path.join(output_dir,  f'{tumor_type_name}_coefficients.xlsx')}')

    return (df_concatenated)


def statistic_from_coefficients (Coefficients_df:pd.DataFrame, true_class: list):
    """ 
    Args: 
        Coefficients_df: DataFrame that contains all the coefficients of the cross folded Logistic Regression.

    Returns: 
        coefficients_stats: Dataframe with the mean, std, Coeficient of Variation and p_value of the coefficients. Values per each protein. .
        significant_proteins: List of significant proteins, sorted by coefficient value.
    """
    #Reading file and obtaining statistics from coefficients
    coefficients_stats = Coefficients_df.describe()
    coefficients_stats = coefficients_stats.T[coefficients_stats.loc['mean']!=0].T[1:3] #This selecst coffiecients different than 0 and the rows 'mean and sd'

    #Calculating frequency
    coefficients_stats.loc['Freq'] = [(Coefficients_df[column] != 0).sum()/ Coefficients_df[column].count() for column in coefficients_stats.columns]#-> Selecting by index

    # Calculating Wald Test for each coefficient 
    coefficients_stats.loc['Wald Chi-Square'] = (np.square(coefficients_stats.loc['mean']))/(np.square(coefficients_stats.loc['std']))
    p_values = 1 - chi2.cdf(coefficients_stats.loc['Wald Chi-Square'], df=1)

    fdr_corrections = fdrcorrection(p_values, alpha=.01, method='indep', is_sorted=False)
    coefficients_stats.loc[' p-value_corrected'] = fdr_corrections[1]
    coefficients_stats.loc['Significant'] = fdr_corrections[0] #Significance is definded with alpha = 0.01 from chi2 dist.

    #Defining if the Wald value is greater than the significance level at 99% = 6.635. N
    #coefficients_stats.loc['Significant'] = [True if i > 6.635 else False for i in coefficients_stats.loc['Wald Chi-Square'] ]
    
    coefficients_stats= coefficients_stats.transpose()
    
    #returning list of significant proteins in order of coefficient value
    protein_coefficients_stats = coefficients_stats.iloc[:-5,:].sort_values(by='mean', ascending=False) #Sort by mean value of coefficients

    significant_proteins = protein_coefficients_stats[protein_coefficients_stats['Significant']==1].index.tolist()
    
    ## EXPORTING ## 
    class_name = "_".join(true_class)

    coefficients_stats.to_excel(os.path.join(output_dir, f'{class_name}_folds_stats.xlsx'), engine='xlsxwriter', index=True)

    with open(os.path.join(output_dir,f'{class_name}_selected_features.txt'), 'w') as file:
        for item in significant_proteins:
            file.write("%s\n" % item)


    ## Message to user ##
    print("With ",protein_coefficients_stats.shape[0], " folds, the following statistics were obtained, from feature selection:")
    print("• Mean MCC score:",np.round(coefficients_stats.loc['MCC_score']['mean'], 4), '±', np.round(coefficients_stats.loc['MCC_score']['std'], 4))
    print()
    
    print("--"*20)
    print("• Top 3 proteins with highest coefficients:")
    print(protein_coefficients_stats.head(3))
    print()
    
    print("--"*20)
    print("• List of significant proteins:",significant_proteins)
    print(f"• Number of significant proteins: {len(significant_proteins)}")
    print()
    
    print("--"*20) 
    if np.round(coefficients_stats.loc['MCC_score']['mean'], 4) < 0.70:
        print(" ✖ Warning! ✖: The mean MCC score is below 0.7, indicating poor model performance.")
    elif 0.80 > np.round(coefficients_stats.loc['MCC_score']['mean'], 4) > 0.70:
        print("✦ The mean MCC score is above 0.7, indicating good model performance. ✦")
    else:
        print("★ The mean MCC score is above 0.8, indicating reliable model performance. ★")

    return coefficients_stats, significant_proteins


def reshape_df_for_fitting(training_df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Input DataFrame must have been generated with the `binary_labeling` function.
    Reshape the ARMS training DataFrame to include only the specified proteins.

    Parameters:
    training_df (pd.DataFrame): The DataFrame containing training data.
    selected_features (list): List of proteins to include in the reshaped DataFrame.
    
    Returns:
    pd.DataFrame: Reshaped DataFrame with specified proteins.
    """
    return pd.concat([training_df.iloc[:,:3],training_df.filter(items=selected_features)],axis=1)


def nested_cross_validation_logistic_regression(train_df:pd.DataFrame, n_splits:int, random_state=93, classified_by='code_oncotree'):
    """
    Perform nested cross-validation for Logistic Regression with hyperparameter tuning using GridSearchCV.

    Parameters:
    - train_df: DataFrame containing the training data with features and target variable.
    - random_state: Random state for reproducibility.
    Returns:
    - outer_scores: List of scores from the outer cross-validation.
    - best_params: List of best hyperparameters from the inner cross-validation.
    """

    y = train_df['Classifier']  # True values (dependent variable)
    X = train_df.drop(columns=['Sample name', 'Classifier', classified_by], axis=1) # Independent variables (proteins)

    # Define the hyperparameter grid for Logistic Regression
    param_grid = {'C': [0.1, 1, 10]}

    # Set up the Logistic Regression model with L2 regularization (Ridge)
    logreg = LogisticRegression(penalty='elasticnet',
                                solver='saga',
                                l1_ratio=0,  # Ridge regularization
                                max_iter=10000,
                                class_weight='balanced',
                                warm_start=False,
                                random_state=93)

    # Set up MCC scorer
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Set up inner and outer cross-validation
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store results
    outer_scores = []
    best_params = []
    inner_fold_cycle = 1
    # Perform nested cross-validation
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner cross-validation with GridSearchCV
        grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=inner_cv, scoring=mcc_scorer)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ConvergenceWarning)
            grid_search.fit(X_train, y_train)

            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    print(f"Inner fold model did not converged.")

        # Get the best parameters and the score for the inner CV
        best_param = grid_search.best_params_
        best_score = grid_search.best_score_

        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        outer_score = matthews_corrcoef(y_test, y_pred)

        # Store the results
        outer_scores.append(outer_score)
        best_params.append(best_param)

        # Print the best parameters and the cross-validation score for each fold
        print(f"{inner_fold_cycle} Inner fold best parameter={best_param}, Score={best_score:.4f}, Outer Validation MCC Score: {outer_score:.4f}")
        print()
        inner_fold_cycle += 1
 
    # Print overall mean MCC score
    print(f"Average MCC across all outer folds: {np.mean(outer_scores):.4f}")
    
    return outer_scores, best_params


def wrapper_nested_cv(train_df:pd.DataFrame, random_state_tries=4, n_splits=4, classified_by='code_oncotree') -> dict:
    """
    Wrapper function for nested cross-validation. 
    Repeats the nested cross-validation process for a specified number of random state tries.

    Args:
        train_df (pd.DataFrame): The training DataFrame.
        random_state_tries (int, optional): Number of random state tries. Defaults to 4.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 4.
        classified_by (str, optional): Column name used for classification. Defaults to 'code_oncotree'.

    Returns:
        dict: Results of the nested cross-validation.
    """
    
    results = {}
    for i in list(range(random_state_tries)):
        print(f"• Running for random_state={i}")
        outer_scores, best_params = nested_cross_validation_logistic_regression(
            train_df, 
            random_state=i,
            n_splits=n_splits,
            classified_by=classified_by
        )
        results[i] = {
            'outer_scores': outer_scores,
            'best_params': best_params
        }
        print()
        print("-" * 50)
    
    return results


def nested_cv_hparameters_selection (input_dict:dict):
    
    # Initialize the result dictionary
    result = {}

    # Loop through the original dictionary
    for key, value in input_dict.items():
        scores = value['outer_scores']
        params = value['best_params']
        
        for score, param in zip(scores, params):
            C_value = param['C']
            
            # If the C value is not already in the result, initialize it
            if C_value not in result:
                result[C_value] = {'scores': [], 'count': 0, 'avg': 0}
            
            # Append the score, update count
            result[C_value]['scores'].append(score)
            result[C_value]['count'] += 1

    # Calculate averages
    for C_value, stats in result.items():
        stats['avg'] = sum(stats['scores']) / stats['count']
    print (pd.DataFrame(result))
    return result


