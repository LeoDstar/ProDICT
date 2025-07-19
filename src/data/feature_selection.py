import pandas as pd
import numpy as np



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