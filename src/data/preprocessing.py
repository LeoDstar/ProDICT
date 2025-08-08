# Function to preprocess the input files from TOPAS pipeline for ML modelling

### Dependencies ###
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import importlib
import sys

### Paths ###
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
output_dir = os.path.join(project_root,'tumor_type_prediction', 'data', 'data_output','data_frames')
os.makedirs(output_dir, exist_ok=True)

### Functions ###

def read_table_with_correct_sep(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".tsv", ".txt"]:
        return pd.read_csv(file_path, sep="\t")
    elif ext == ".xlsx":
        return pd.read_excel(file_path, engine='openpyxl')

    else:
        raise ValueError(f"Unsupported file extension for : {file_path}")

def post_process_meta_intensities(intensity_meta:pd.DataFrame) -> pd.DataFrame : 
    intensity_meta = intensity_meta.fillna('num_peptides=0;')
    intensity_meta = intensity_meta.replace('num_peptides=|;','',regex=True)
    intensity_meta = intensity_meta.replace('detected in batch','0', regex=True) 
    intensity_meta = intensity_meta.apply(pd.to_numeric, errors='coerce')
   
    return intensity_meta

def impute_normal_down_shift_distribution(unimputerd_dataframe:pd.DataFrame ,column_wise=True, width=0.3, downshift=1.8, seed=2):
    """ 
    Performs imputation across a matrix columnswise
    https://rdrr.io/github/jdreyf/jdcbioinfo/man/impute_normal.html#google_vignette
    
    Data imputation to replace NaN Values, code taken from PERSEUS
    
    :width: Scale factor for the standard deviation of imputed distribution relative to the sample standard deviation.
    :downshift: Down-shifted the mean of imputed distribution from the sample mean, in units of sample standard deviation.
    :seed: Random seed
    
    """
    
    print(unimputerd_dataframe.shape)

    unimputerd_df = unimputerd_dataframe.apply(pd.to_numeric, errors="coerce")
    if unimputerd_df.shape[1] == 0:
        raise ValueError("No numeric columns found in the input dataframe for imputation.")

    unimputerd_matrix = unimputerd_df.to_numpy(dtype=np.float64)
    unimputerd_matrix[~np.isfinite(unimputerd_matrix)] = np.nan

    columns_names = unimputerd_df.columns
    rownames = unimputerd_df.index

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
        return temp
    
    final_matrix = np.apply_along_axis(impute_normal_per_vector, 0, unimputerd_matrix)
    final_df = pd.DataFrame(final_matrix)
    final_df.index = rownames
    final_df.columns = columns_names
    
    return final_df

def remove_class(df: pd.DataFrame, class_list: list, classified_by: str) -> pd.DataFrame:
    """
    Remove rows from DataFrame that are unnecesary for training, and export the removed samples to a CSV file.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        class_list (list): List of class labels to be removed.
        classification_by (str): Column name in the DataFrame that contains the class labels.
    Returns:
        pd.DataFrame: DataFrame with specified classes removed.
    """

    modified_df = df[~df[classified_by].isin(class_list)]

    removed_samples = df[df[classified_by].isin(class_list)]
    removed_samples.to_excel(os.path.join(output_dir,'removed_samples_nos.xlsx'), engine='xlsxwriter', index=False)

    print(f"Removed samples: {len(removed_samples)}")
    print(f"Remaining samples: {len(modified_df)}")
        
    return modified_df.reset_index(drop=True)

def data_split(df: pd.DataFrame, split_size=0.25, classified_by='code_oncotree', export=True) -> tuple[pd.DataFrame, pd.DataFrame]: 
    """
    Stratified split of the dataset into training and held-out sets, ensuring that each class has at least two samples. from scikit-learn

    Parameters:
    df (pd.DataFrame): The initial DataFrame containing the data to be split.
    
    Returns:
    X_train, X_held_out, y_train, y_held_out: The training and held-out sets.
    """
    # Filter out classes with only one sample
    df_wo_single_cases = df.groupby(classified_by).filter(lambda x: len(x) > 1)
    print ("Classes with only one sample:", df.groupby(classified_by).filter(lambda x: len(x) == 1).shape[0])

    #stratified splitting using scikit-learn
    X_train, X_held_out, y_train, y_held_out = train_test_split(
                                                                df_wo_single_cases.drop(columns=['Sample name', classified_by]), 
                                                                df_wo_single_cases[classified_by], 
                                                                test_size=split_size, 
                                                                stratify=df_wo_single_cases[classified_by], 
                                                                random_state=1,
                                                                )
        
    #Generate training and held-out DataFrames
    training_indices = list(X_train.index) + list(df.groupby(classified_by).filter(lambda x: len(x) == 1).index)
    training_df_ = df.loc[training_indices]
    training_df_.sort_index(inplace=True)
    print(f"Training set samples: {len(training_indices)}")

    held_out_indices = list(set(X_held_out.index) - set(training_indices))
    held_out_df = df.loc[held_out_indices]
    held_out_df.sort_index(inplace=True)
    print(f"Held-out set samples: {len(held_out_indices)}")
    
    if export:
        training_df_.to_excel(os.path.join(output_dir, 'initial_training_df.xlsx'), engine='xlsxwriter', index=False)
        held_out_df.to_excel(os.path.join(output_dir, 'held_out_df.xlsx'), engine='xlsxwriter', index=False)

    return training_df_, held_out_df

