
#-------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#---------------------------------------------------------
## DATASET CLEANING ##
#---------------------------------------------------------

def clean_df (intensity_df:pd.DataFrame, metadata_df:pd.DataFrame ):
    
    intensity_df = intensity_df.transpose()
    intensity_df.reset_index(inplace=True)
    intensity_df = intensity_df.rename(columns=intensity_df.iloc[0]).drop([0])

    #return intensity_df
    metadata_df = metadata_df[["Sample name", "code_oncotree", ]] #Selection of columns for later concatenate

    df_merged = metadata_df.merge(intensity_df, left_on='Sample name', right_on='Gene names') #merging both data sets by Sample Name
    df_merged.drop('Gene names', axis=1, inplace=True)
    return df_merged



#--------------------------------------------------------------------------
#Data imputation to replace NaN Values, code taken from PERSEUS
def impute_normal_down_shift_distribution(unimputerd_dataframe:pd.DataFrame ,column_wise=True, width=0.3, downshift=1.8, seed=2):
    """ 
    Performs imputation across a matrix columnswise
    https://rdrr.io/github/jdreyf/jdcbioinfo/man/impute_normal.html#google_vignette
    :width: Scale factor for the standard deviation of imputed distribution relative to the sample standard deviation.
    :downshift: Down-shifted the mean of imputed distribution from the sample mean, in units of sample standard deviation.
    :seed: Random seed
    
    """
    
    print(unimputerd_dataframe.shape)

    unimputerd_df = unimputerd_dataframe.iloc[:,:]

    unimputerd_matrix = unimputerd_df.replace({pd.NA: np.nan}, inplace=True) #Added to modify pandas's NAN values into  numpy NAN values
    
    unimputerd_matrix = unimputerd_df.to_numpy()
    columns_names = unimputerd_df.columns
    rownames = unimputerd_df.index
    unimputerd_matrix[~np.isfinite(unimputerd_matrix)] = None
    main_mean = np.nanmean(unimputerd_matrix)
    main_std = np.nanstd(unimputerd_matrix)
    np.random.seed(seed = seed)
    def impute_normal_per_vector(temp:np.ndarray,width=width, downshift=downshift):
        """ Performs imputation for a single vector """
        if column_wise:
            temp_sd = np.nanstd(temp)
            temp_mean = np.nanmean(temp)
        else:
            # over all matrix
            temp_sd = main_std
            temp_mean = main_mean

        shrinked_sd = width * temp_sd
        downshifted_mean = temp_mean - (downshift * temp_sd) 
        n_missing = np.count_nonzero(np.isnan(temp))
        temp[np.isnan(temp)] = np.random.normal(loc=downshifted_mean, scale=shrinked_sd, size=n_missing)
        if n_missing > 0:
            print 
        return temp
    final_matrix = np.apply_along_axis(impute_normal_per_vector, 0, unimputerd_matrix)
    final_df = pd.DataFrame(final_matrix)
    final_df.index = rownames
    final_df.columns = columns_names
    
    return final_df
#--------------------------------------------------------------
#Definition of the Sigmoid!! Function  
def logistic_function(x):    
    return 1/ (1 + np.exp(-x))

#--------------------------------------------------------------
#DROPS PROTEINS COLUMNS WITH FULL NAN VALUES 
def NaNs_remover (imptd_df:pd.DataFrame):

   NaN_vals = imptd_df.describe().isna().loc['mean'][imptd_df.describe().isna().loc['mean'] == True].index.tolist()
   print (f' Proteins with NaN values that were removed are : {NaN_vals}')
   imptd_df.dropna(axis=1, inplace=True)
    
   return imptd_df


#------------------------------------------------------
## DATASET Processsing ##
#------------------------------------------------------

#---------------------------------------------------------------   
#Calculates the mean, std, normal distribution and Wald Chi Test, for all the coefficients generated. 
from statsmodels.stats.multitest import fdrcorrection

def coeff_stats (Coefficients_df:pd.DataFrame):
    """ 
    Args: input pandas Dataframe that contains all the coefficients of the Logistic Regression.

    Returns:1.- Dataframe with the mean, std, Coeficient of Variation and p_value of the coefficients. Values per each protein. 
    """
    #Reading file and obtaining statistics from coefficients
    entity= Coefficients_df.describe()
    entity_stats = entity.T[entity.loc['mean']!=0].T[1:3] #This selecst coffiecients different than 0 and the rows 'mean and sd' 
    
    
    #Calculating frequency 
    entity_stats.loc['Freq'] = [(Coefficients_df[column] != 0).sum()/ Coefficients_df[column].count() for column in entity_stats.columns]#-> Selecting by index

    #Calculating p_value for each column(protein) using Kolmogorov-Smirnov function and accesing p_value form the lists created
    #entity_stats.loc['Kolmogorov'] = [i[1] for i in [kstest_normal(Coefficients_df[column], pvalmethod='approx') for column in entity_stats.columns]]

    # Calculating Wald Test for each coefficient 
    entity_stats.loc['Wald Chi-Square'] = (np.square(entity_stats.loc['mean']))/(np.square(entity_stats.loc['std']))

    p_values = 1 - chi2.cdf(entity_stats.loc['Wald Chi-Square'], df=1)

    fdr_corrections = fdrcorrection(p_values, alpha=.01, method='indep', is_sorted=False)
    entity_stats.loc[' p_val_corrected'] = fdr_corrections[1]
    entity_stats.loc['Significant'] = fdr_corrections[0] #Significance is definded with alpha = 0.01 from chi2 dist.

    #Defining if the Wald value is greater than the significance level at 99% = 6.635. N
    #entity_stats.loc['Significant'] = [True if i > 6.635 else False for i in entity_stats.loc['Wald Chi-Square'] ]
    
    entity_stats = entity_stats.transpose()
    
    return entity_stats
    
    
#----------------------------------------------------------------------------------------------------
#Calculates the scores of classification by different methods. NOT USED - REPLACED BY Classification_Scores
def Test_Group_Scores (df, df_coefficients, entity_code: str,):

    df['Classifier'] = np.where(df['code_oncotree'] == entity_code, 1, 0) #Creating a binary label based on oncotree,change it for each code
    y = df['Classifier'] #entity 
    X = df.iloc[:,:-1] #protein
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=93)

    y_test_predict = np.round(logistic_function( np.dot([X_test], coeff_CHDM01[:-6]) + intercept_CHDM01 ))
    
    MCC = matthews_corrcoef(y_test, y_test_predict[0])
    F1_weighted = f1_score(y_test, y_test_predict[0], average='weighted')
    F1_1 = f1_score(y_test, y_test_predict[0], pos_label=1)
    #ROC_AUC = roc_auc_score(y_test, y_test_predict[0])
    
    print (f'MCC Score: {MCC}')
    print (f'F1 Macro Score: {F1_Macro}')
    print (f'F1 Entity Score: {F1_1}')
    #print (f'ROC-AUC Score: {ROC_AUC}')
#---------------------------------------------------------------------------
#Calculates the probabilities of being True or False for each sample

def entity_probabilities_calcualtion (df_clean , df_coefficients):
    """
    df_clean: Dataframe used for calculate coefficients. Initial dataset, with  Sample name	+ code_oncotree + proteins intensities. and imputated, or removed NaN values 
    
    df_coefficients: result from the coefficients analysis, obtained from the function "log_reg_results_stats"
    
    """
    df = df_clean.copy()
        
    
    coeff = np.array([0.0] * df.shape[0])
    
    for i in df_coefficients.iloc[:,:-5].columns:
        try:
            coeff += np.dot(df[i],df_coefficients.loc['mean'][i])
            
        except KeyError:
            print(f"Protein {i} not found. Check its intensity")
            continue
    
        coeff_sum = coeff
    
    
    probability = logistic_function( coeff_sum + df_coefficients.loc['mean', 'Intercept'] )

    df.insert(2, "Probability", probability)
    df.iloc[:,:3].to_excel('entity_patients_probabilities.xlsx') #used to export file as excel

    
    return df.iloc[:,:3]

#------------------------------------------------------------
#Calculates the classification scores by diffent methods
def Classification_Scores (calc_probs_df, entity_code= ''):
    """Calculates the classification scores from the prediction values with a threshold of 0.5. 
    
    Inputs: 
    calc_probs_df : 
    entity_code : 
    Args:
        calc_probs_df : Dataframe with the calculated probabilities.
        entity_code (str, optional): Code for the entity to evaluate. Defaults to ''.
    """

    # Creating labesl based on predictions
    probabilities_df = calc_probs_df.copy()
    probabilities_df.reset_index(drop=True, inplace=True)
    probabilities_df['Predicted'] = np.rint(probabilities_df['Probability'])
    probabilities_df['True'] = np.where(probabilities_df['code_oncotree'] == entity_code, 1, 0) #Creating a binary label based oncotree,change it for each code

    # Perdormance Scores -----------------------------------------------------------------------
    MCC = matthews_corrcoef(probabilities_df['True'], probabilities_df['Predicted'])
    F1_Macro = f1_score(probabilities_df['True'], probabilities_df['Predicted'], average='macro') #For Imbalanced Dataset
    F1_Micro = f1_score(probabilities_df['True'], probabilities_df['Predicted'], average='micro') #For Balanced Datasets
    F1_1 = f1_score(probabilities_df['True'], probabilities_df['Predicted'], pos_label=1)
    ConfusionMatrix = confusion_matrix(probabilities_df['True'], probabilities_df['Predicted'])

    # False positives detection -----------------------------------------------------------------
    False_Positives = []
    for i in range(probabilities_df.shape[0]): 
        if (probabilities_df['Predicted'][i] == 1) & (probabilities_df['True'][i] == 0):
            False_Positives.append(probabilities_df.loc[i])

    # False Negatives Detection -----------------------------------------------------------------
    False_Negatives = []
    for i in range(probabilities_df.shape[0]): 
        if (probabilities_df['Predicted'][i] == 0) & (probabilities_df['True'][i] == 1):
            False_Negatives.append(probabilities_df.loc[i])


    
    print (f'MCC Score: {MCC}')
    print (f'F1 Macro: {F1_Macro}')
    print (f'F1 Micro: {F1_Micro}')
    print (f'F1 Entity Score: {F1_1}')
    print ('------------------------------------')
    print (ConfusionMatrix)
    print ('------------------------------------')
    print ('False Positives:')
    print (pd.DataFrame(False_Positives))
    print ('------------------------------------')
    print ('False Negatives:')
    print (pd.DataFrame(False_Negatives))


    return probabilities_df


#-------------------------------------------------------------
#LOGISTIC REGRESSION FUNCTIONS
#-------------------------------------------------------------
#Logit Function 
#Logistic Regression with 5CrossValidation

def ml_log_reg_5cv (df:pd.DataFrame, X, r=1):
    """
    This function return the coefficients of the independent variables defined by ML Logistic Regression. Uses 5k Cross Validation. 
    df:input the imputated and concatenated dataframe, generated with previous functions.
    X: provide the name of the entity as a string. e.g. "CHDM" or "ARMS"
    r: random state of CV

    """

    log_reg_coeff_list = []
    results = []
  
    #Selecting Data
    

    y = df['Classifier'] #entity 
    X = df.iloc[:,2:-1] #protein
            
        
    #First Data Splitting ; Stratified for entity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=93 )
        
        
    #Defining model parameters
    log_reg = LogisticRegression(penalty='elasticnet',  
                                 solver='saga', 
                                 l1_ratio=0.5 ,
                                 max_iter=7000, 
                                 C = 1, 
                                 class_weight= 'balanced',
                                 warm_start=True,
                                )
    #Saving the id used for each fold:
        
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    p = 1 
    for train_index, test_index in skf.split(X_train, y_train):
        x_train_fold, x_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        log_reg.fit(x_train_fold, y_train_fold)
        
    #Logaritmic regression model coeffiecient:
        log_reg_coefficients = log_reg.coef_
        log_reg_coeff_list = log_reg_coefficients.tolist()
    
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
   
        print(p)
        p +=1
    
    col_names = list(df.iloc[:,2:-1].columns)
    col_names.extend(['Intercept','F1_1', 'F1_0','F1_weighted','MCC_score' ])
    
    
    coeff_score_df = pd.DataFrame(results)
    coeff_score_df.columns = col_names
    
    return(coeff_score_df)


#--------------------------------------------------------------------

def ml_log_reg_coefficients (df:pd.DataFrame, X, r=1):
    
    """_summary_ 

    Returns: Returns a dataframe with the coefficients of Logistic Regression
    df: Input dataframe with proteins intensities, imputated and with no NaN values
    r: input the number of repetitions. r = 1 : 1 experiment with 5k Cross validation 
        _type_: _description_
    """
        
    df['Classifier'] = np.where(df['code_oncotree'] == X, 1, 0) #Creating a binary label based on oncotree,change it for each code
    tries = range(0,r) #number of times the ML algorithm will run. triesx5 = #coefficients
    final_df_list = []
    
    progress = 1
    
    for i in tries:
        print ('-', progress , '-')
        progress += 1
        
        final_df_list.append(ml_log_reg_5cv(df,X, i))
        
    
    df_concatenated = pd.concat(final_df_list, ignore_index=True) #keys=["fold1", "fold2"...] if needed to specify the fold results
    
    #File name
    excel_file_path = f'{X}_coefficients.xlsx'
    
    # Export the DataFrame to an Excel file
    df_concatenated.to_excel(excel_file_path, index=False)
    
    print(f'DataFrame exported as {excel_file_path}')
    return (df_concatenated)

#---------------------------------------------------------
def export_list_to_txt(my_list):
    """
    Export a Python list to a text file.

    Parameters:
        my_list (list): The list to export.
        file_path (str): The path to the output text file.
    """
    # Open the file in write mode
    with open(f'{my_list}.txt', 'w') as file:
        # Write each item in the list to the file
        for item in my_list:
            file.write("%s\n" % item)

    print("List exported as :", f'{my_list}.txt')
    

#---------------------------------------------------------

def Log_Reg_regularized(df: pd.DataFrame, entity) -> pd.DataFrame:
    
    """Logistic Regression, regularized with Ridge. 
    Input: 
    df: Input the clean dataframe, Sample name + Classification + Protein Intensities. 
    entity: specify the code of the entity to create the model. 
    
    Returns:
        coeff_score_df: Coefficients calculated + Intercept + Scores from the Logistic Regression 
        prob_ALL: Dataframe with the probablities for each sample of the whole data set
        prob_HO: Dataframe with the probablities for each sample of the HELD OUT group
    """
    df['Classifier'] = np.where(df['code_oncotree'] == entity, 1, 0) #Creating a binary label based on oncotree,change it for each code
    y = df['Classifier'] #entity 
    X = df.iloc[:,2:-1] #protein
    results =[]        
        
    #First Data Splitting ; Stratified for entity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=93 )
        
        
    #Defining model parameters
    log_reg = LogisticRegression(penalty='elasticnet',  
                                 solver='saga', 
                                 l1_ratio=0 , #Ridge regularization
                                 max_iter=50000, 
                                 C = 1, 
                                 class_weight= 'balanced',
                                 warm_start=True,
                                )
    
    log_reg.fit(X_train, y_train)
    
    #Logaritmic regression model coeffiecient:
    log_reg_coefficients = log_reg.coef_
    log_reg_coeff_list = log_reg_coefficients.tolist()
    
    #Logaritmic regression model coeffiecient INTERCEPT:
    log_reg_coeff_list[0].append (log_reg.intercept_[0])
        
    #Model predictions for score in folds
    y_predict = log_reg.predict(X_test)
    y_train_predict = log_reg.predict(X_train)
         
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
    
    #MCC Score
    MCC_score = matthews_corrcoef(y_test, y_predict)
    log_reg_coeff_list[0].append(MCC_score)

    #MCC Score Train
    MCC_score_train = matthews_corrcoef(y_train, y_train_predict)
    
    
    
    col_names = list(df.iloc[:,2:-1].columns)
    col_names.extend(['Intercept','F1_1', 'F1_0','F1_weighted','MCC_score' ])
    
    
    coeff_score_df = pd.DataFrame(results)
    coeff_score_df.columns = col_names
    
    print('# of Iterations:',log_reg.n_iter_)
    print('MCC train:',MCC_score_train)
    print('MCC test:',MCC_score)
    print('F1 Positive:',f1_score_class_1)

    prob_HO = df.iloc[X_test.index, :2]
    prob_HO.insert(2, value = log_reg.predict_proba(X_test).T[1] , column = 'Probability')
    prob_HO
    prob_HO_file_path = f'{entity}_prob_HO.xlsx'
    prob_HO.to_excel(prob_HO_file_path, index=False)

    
    prob_ALL = df.iloc[:,:2]
    prob_ALL.insert(2, value = log_reg.predict_proba(df.iloc[:,2:-1]).T[1] , column = 'Probability')
    prob_ALL
    prob_ALL_file_path = f'{entity}_prob_ALL.xlsx'
    prob_ALL.to_excel(prob_ALL_file_path, index=False)
    
    return (coeff_score_df, prob_ALL , prob_HO, log_reg)  

#--------------------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
def create_pairplot(df, proteins=[], entity=''):

    # Assign values to the 'Classifier' column based on the condition
    data = df[proteins]
    data['Classifier'] = np.where(df['code_oncotree'] == entity, f'{entity}', f'Non-{entity}')
    # Plot the pairplot
    sns.set_style("white")
    sns.pairplot(data, hue='Classifier', corner=True, palette=['#448AFF','#8BC34A'])
    graph_name = f'{entity}_pairplot.png'
    plt.savefig(graph_name,dpi=600) 
    plt.show()


#--------------------------------------------------------------------------------------------

def log_likelihood(df:pd.DataFrame, entity = ''):
    """
    Calculate the log-likelihood given the true class labels and predicted probabilities.

    Parameters:
    - y_true: Array-like, true class labels (0 or 1).
    - y_pred: Array-like, predicted probabilities of the positive class.

    Returns:
    - log_likelihood: Log-likelihood value.
    """
    df['y'] = np.where(df['code_oncotree'] == entity, 1,0)
    log_likelihood = np.sum(np.log(np.where(df['y'] == 1, df['Probability'], 1 - df['Probability'])))
    
    return log_likelihood
