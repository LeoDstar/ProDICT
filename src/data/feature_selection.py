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