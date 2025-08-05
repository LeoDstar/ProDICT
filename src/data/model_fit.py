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
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm  
import warnings
from sklearn.exceptions import ConvergenceWarning


### Paths ###
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
output_dir = os.path.join(project_root, 'data', 'data_output')
os.makedirs(output_dir, exist_ok=True)


### Functions ###

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
                                warm_start=False)

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


def log_likelihood(df:pd.DataFrame):
    """
    Calculate the log-likelihood given the true class labels and predicted probabilities.

    Parameters:
    - y_true: Array-like, true class labels (0 or 1).
    - y_pred: Array-like, predicted probabilities of the positive class.

    Returns:
    - log_likelihood: Log-likelihood value.
    """
    log_likelihood = np.sum(np.log(np.where(df['Classifier'] == 1, df['Probability'], 1 - df['Probability'])))
    
    return log_likelihood


def classification_scores (calc_probs_df):
    """Calculates the classification scores from the prediction values with a threshold of 0.5. 
    
    Inputs: 
    calc_probs_df : 

    Args:
        calc_probs_df : Dataframe with the calculated probabilities.

    """

    # Creating labesl based on predictions
    probabilities_df = calc_probs_df.copy()
    probabilities_df.reset_index(drop=True, inplace=True)
    probabilities_df['Predicted'] = np.rint(probabilities_df['Probability'])
    
    # Perdormance Scores -----------------------------------------------------------------------
    MCC = matthews_corrcoef(probabilities_df['Classifier'], probabilities_df['Predicted'])
    F1_Macro = f1_score(probabilities_df['Classifier'], probabilities_df['Predicted'], average='macro') #For Imbalanced Dataset
    F1_Micro = f1_score(probabilities_df['Classifier'], probabilities_df['Predicted'], average='micro') #For Balanced Datasets
    F1_1 = f1_score(probabilities_df['Classifier'], probabilities_df['Predicted'], pos_label=1)
    ConfusionMatrix = confusion_matrix(probabilities_df['Classifier'], probabilities_df['Predicted'])

    # False positives detection -----------------------------------------------------------------
    False_Positives = []
    for i in range(probabilities_df.shape[0]): 
        if (probabilities_df['Predicted'][i] == 1) & (probabilities_df['Classifier'][i] == 0):
            False_Positives.append(probabilities_df.loc[i])

    # False Negatives Detection -----------------------------------------------------------------
    False_Negatives = []
    for i in range(probabilities_df.shape[0]): 
        if (probabilities_df['Predicted'][i] == 0) & (probabilities_df['Classifier'][i] == 1):
            False_Negatives.append(probabilities_df.loc[i])


    print ('------------------------------------')
    print ('•General Scores:')
    print (f'MCC Score: {MCC}')
    print (f'F1 Macro: {F1_Macro}')
    print (f'F1 Micro: {F1_Micro}')
    print (f'F1 Entity Score: {F1_1}')
    print()
    print ('------------------------------------')
    print ('•Confusion Matrix:')
    print ('  TN | FP')
    print (ConfusionMatrix)
    print ('  FN | TP')
    print()
    print ('------------------------------------')
    print ('•False Positives:')
    if False_Positives:
        print (pd.DataFrame(False_Positives))
    else:
        print("No False Positives detected.")   
    print()
    print ('------------------------------------')
    print ('•False Negatives:')
    if False_Negatives:
        print (pd.DataFrame(False_Negatives))
    else:
        print("No False Negatives detected.")


    return probabilities_df


def logistic_regression_results (log_reg_model, df_train: pd.DataFrame, df_test: pd.DataFrame, true_class:list, classified_by:str ) -> tuple:  
    """
    class_criteria: string that defines the classification criteria (i.e. 'Tissue_origin' or 'tissue_topology'). 'code_oncotree' is always included. 
    """
    ### Data preparation ###
    results =[]  
    class_name = "_".join(true_class)

    X_train = df_train.drop(columns=['Sample name', 'Classifier', classified_by], axis=1)
    y_train = df_train['Classifier']

    X_test = df_test.drop(columns=['Sample name', 'Classifier', classified_by], axis=1)
    y_test = df_test['Classifier']

    ### Processing Data from model ###
    #Logaritmic regression model coeffiecient:
 
    log_reg_coeff_list = log_reg_model.coef_.tolist()
    
    #Logaritmic regression model coeffiecient INTERCEPT:
    log_reg_coeff_list[0].append (log_reg_model.intercept_[0])
        
    #Model predictions for score in folds
    y_predict = log_reg_model.predict(X_test.filter(items=log_reg_model.feature_names_in_))
    y_train_predict = log_reg_model.predict(X_train)
         
    results.append(log_reg_coeff_list[0])
    
    #F1_score for class 1 - entity prediction
    f1_score_class_1 = f1_score(y_test, y_predict, pos_label=1)
    log_reg_coeff_list[0].append(f1_score_class_1)
    
    #F1_score for class 0 - entity prediction
    f1_score_class_0 = f1_score(y_test, y_predict, pos_label=0)
    log_reg_coeff_list[0].append(f1_score_class_0)
    
    #F1_score for class 1 - entity prediction
    f1_score_class_weighted = f1_score(y_test, y_predict, average= 'weighted')
    log_reg_coeff_list[0].append(f1_score_class_weighted)
    
    #MCC Score test
    MCC_score_test = matthews_corrcoef(y_test, y_predict)
    log_reg_coeff_list[0].append(MCC_score_test)

    #MCC Score Train
    MCC_score_train = matthews_corrcoef(y_train, y_train_predict)


    col_names = list(X_train.columns)
    col_names.extend(['Intercept','F1_1', 'F1_0','F1_weighted','MCC_score' ])
    
    final_model_coefficients = pd.DataFrame(results)
    final_model_coefficients.columns = col_names
    
    ### Results of model fit ###
    print(f'# of Iterations: {log_reg_model.n_iter_}')
    print(f'MCC train: {MCC_score_train}')
    print(f'MCC test: {MCC_score_test}')
    print(f'F1 Positive: {f1_score_class_1}')



    test_probabilities = df_test[['Sample name','code_oncotree' ,'Classifier']]
    test_probabilities.insert(3, value = log_reg_model.predict_proba(df_test.filter(items=log_reg_model.feature_names_in_)).T[1] , column = 'Probability')

    test_probs_file = os.path.join(output_dir,f'test_{class_name}_probabilities.xlsx')
    test_probabilities.to_excel(test_probs_file, index=False)

    
    train_probabilities = df_train[['Sample name','code_oncotree','Classifier']]
    train_probabilities.insert(3, value = log_reg_model.predict_proba(df_train.filter(items=log_reg_model.feature_names_in_)).T[1] , column = 'Probability')
    

    train_probs_path = os.path.join(output_dir,f'train_{class_name}_probabilities.xlsx')
    train_probabilities.to_excel(train_probs_path, index=False)
    
    return (final_model_coefficients, train_probabilities , test_probabilities)  


def logistic_regression_ridge(df: pd.DataFrame, C_:float, true_class:list, classified_by:str) -> LogisticRegression : #Define the hypeparameters for this entity

    """Logistic Regression, regularized with Ridge. 
    Input: 
    df: Input TRAINING DATAFRAME, Sample name + Classification + Protein Intensities. 
    entity: specify the code of the entity to create the model. 
    
    Returns:
        SKLEARN model object. This can be used for former analysis.
    """
    #Selecting Data
    y_train = df['Classifier']  # True values (dependent variable)
    X_train = df.drop(columns=['Sample name', 'Classifier', classified_by], axis=1) # Independent variables (proteins)


    #Defining model parameters
    log_reg = LogisticRegression(penalty='elasticnet',  
                                 solver='saga', 
                                 l1_ratio=0, #Ridge regularization
                                 max_iter=10000, 
                                 C = C_, 
                                 class_weight= 'balanced',
                                 warm_start=False,
                                 random_state=93
                                )
    
    log_reg.fit(X_train, y_train)
    
    class_name = "_".join(true_class)
    dump(log_reg, os.path.join(output_dir,f'{class_name}_log_reg_ridge_model.pkl'))
    print(f'Model saved as {class_name}_log_reg_ridge_model.pkl')

    return log_reg   



