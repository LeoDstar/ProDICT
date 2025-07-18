# Function to preprocess the input files from TOPAS pipeline for ML modelling

import pandas as pd
import numpy as np


def post_process_meta_intensities(intensity_meta:pd.DataFrame) -> pd.DataFrame : 
    intensity_meta = intensity_meta.fillna('num_peptides=0;')
    intensity_meta = intensity_meta.replace('num_peptides=|;','',regex=True)
    intensity_meta = intensity_meta.replace('detected in batch','0', regex=True) 
    intensity_meta = intensity_meta.apply(pd.to_numeric, errors='coerce')
   
    return intensity_meta

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