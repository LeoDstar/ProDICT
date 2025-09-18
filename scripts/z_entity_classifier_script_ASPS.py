"""
Classifier Workflow Script
Command-line runnable version of the original Jupyter notebook
"""

###############
### Imports ###
###############
import sys
import pandas as pd
import numpy as np # type: ignore
import warnings
import logging
import multiprocessing as mp
import sys
import os

from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from sklearn.preprocessing import StandardScaler
import pickle


current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
module_path = project_root / "src" / "data"
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
    
from entity_model_settings_ASPS import *

project_root = os.path.abspath(os.getcwd())
output_dir = os.path.join(project_root, 'data', run_folder_name)
os.makedirs(output_dir, exist_ok=True)

#####################
### Logging Setup ###
#####################
mp.set_start_method('spawn', force=True)  


class TeeOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

output_filename = f"classifier_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
tee_output = TeeOutput(output_filename)

#################
### Functions ###
#################

def setup_paths():
    """Setup module paths for imports"""
    # Get the project root directory (parent of current script directory)
    current_script_dir = Path(__file__).parent.absolute()
    project_root = current_script_dir.parent
    
    # Add src/data to path
    module_path = str(project_root / "src" / "data")
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    return project_root

def import_custom_modules():
    """Import custom modules with error handling"""
    try:
        import preprocessing as prep
        import feature_selection as fs
        import model_fit as mf
        import graphs as grph
        return prep, fs, mf, grph
    except ImportError as e:
        print(f"Error importing custom modules: {e}")
        print("Make sure the following modules are in src/data/:")
        print("- LogRegFxF.py")
        print("- preprocessing.py") 
        print("- feature_selection.py")
        print("- model_fit.py")
        sys.exit(1)

def load_data(prep):
    """Load all required data files"""
    print("="*80)
    print("Loading training and test files...")
    print("="*80)
    print(f"Target class: {TARGET_CLASS}")
    print(f"Classification column: {CLASSIFIED_BY}")
    
    held_out_path = '/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/LE_PROdict/paper_data/held_out_df.xlsx'
    train_path = '/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/LE_PROdict/paper_data/initial_training_df.xlsx'
    target_proteins_path = f"/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/LE_PROdict/paper_classifier_results/04_classifiers_protein_lists/{target_class_name}_selected_features.txt"

    
    try:
        held_out_df = pd.read_excel(held_out_path)
        train_df = pd.read_excel(train_path)
        
        with open(target_proteins_path, 'r') as fd:
            target_proteins = [line.strip() for line in fd if line.strip()]

        print("Data files loaded successfully.")
        print(f"Train Dataset shape: {train_df.shape}")
        print(f"Test Dataset shape: {held_out_df.shape}")


        return held_out_df, train_df, target_proteins

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please check that all data files exist in the specified paths:")
        sys.exit(1)

def preprocess_data(held_out_df, train_df):
    """Preprocess all data"""
    print("="*80)
    print("Preprocessing data, normalizing Train set, and Test set...")
    print("="*80)
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1))
    scaled_train = pd.DataFrame(scaled_train, 
                                columns=train_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1).columns, 
                                index=train_df.index)
    scaled_train = pd.concat([train_df[['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP']], scaled_train], axis=1)
    print("Train dataframe shape:", train_df.shape)
    print("Train Normalized dataframe shape:", scaled_train.shape)

    scaled_test = scaler.transform(held_out_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1))
    scaled_test = pd.DataFrame(scaled_test, 
                               columns=held_out_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1).columns, 
                               index=held_out_df.index)
    scaled_test = pd.concat([held_out_df[['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP']], scaled_test], axis=1)
    print("Test dataframe shape:", held_out_df.shape)
    print("Test Normalized dataframe shape:", scaled_test.shape)
    
    
    with open(f"{output_dir}/{target_class_name}_normalization_parameters.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return scaled_train, scaled_test, scaler

def binary_labelling(training_df, held_out_df, fs):
    """Execute class-specific workflow for specified classification"""
    print("="*80)
    print(f"Labeling positive samples for {TARGET_CLASS}...")
    print("="*80)

    # Binary labeling for specific class classification 
    print("Labeling training dataframe...")
    target_training_df =        fs.binary_labeling(training_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    print("Labeling test dataframe...")
    target_ho_df =              fs.binary_labeling(held_out_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)


    return target_training_df, target_ho_df
 
def model_fitting(target_training_df, target_ho_df, target_proteins, output_directory,fs, mf):
    """Fit the final model and evaluate"""
    print("="*80)
    print("Starting model fitting...")
    print("="*80)
    print(f"Using {NESTED_CV_RANDOM_STATE_TRIES} random state tries for nested CV")
    print(f"Using {NESTED_CV_N_SPLITS} splits for nested CV")
    
    # Reshaping dataset for training and test
    target_training_fs = fs.reshape_df_for_fitting(target_training_df, target_proteins)
    target_test_fs = fs.reshape_df_for_fitting(target_ho_df, target_proteins)
    
    print(f"Training set shape after feature selection: {target_training_fs.shape}")
    print(f"Test set shape after feature selection: {target_test_fs.shape}")

    # Hyperparameter Selection for Logistic Regression
    try:
        target_nested_cv_results = mf.wrapper_nested_cv(
            target_training_fs, 
            random_state_tries=NESTED_CV_RANDOM_STATE_TRIES, 
            n_splits=NESTED_CV_N_SPLITS, 
            classified_by=CLASSIFIED_BY
        )
        target_nested_hp = mf.nested_cv_hparameters_selection(target_nested_cv_results)
        hyperparameter_C = pd.DataFrame(target_nested_hp).T.sort_values(by='count', ascending=False).index.tolist()[0]
        print(f"Selected hyperparameter C: {hyperparameter_C}")
    except Exception as e:
        print(f"Warning: Hyperparameter selection failed: {e}")
        hyperparameter_C = 1.0  # Default value
        print(f"Using default hyperparameter C: {hyperparameter_C}")

    # Model Fit
    try:
        target_log_reg_model = mf.logistic_regression_ridge(
            target_training_fs, 
            hyperparameter_C, 
            TARGET_CLASS, 
            CLASSIFIED_BY,
            output_directory,
            )
        
        # Get results
        target_coefficients, target_train_probabilities, target_test_probabilities = mf.logistic_regression_results(
            target_log_reg_model, 
            target_training_fs, 
            target_test_fs,  
            TARGET_CLASS, 
            CLASSIFIED_BY,
            output_directory
        )
        
        # Classification scores
        test_target_scores = mf.classification_scores(target_test_probabilities)
        
        print("Model training and evaluation completed successfully!")
        return target_log_reg_model, target_coefficients, target_train_probabilities, target_test_probabilities, test_target_scores
        
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None, None, None, None, None

def generate_graphs(initial_df, test_target_scores, target_proteins, output_directory, grph):
    """Generate and save graphs for results exploration"""
    print("="*80)
    print("Generating graphs...")
    print("="*80)
    
    # UMAP plot
    UMAP_plot = grph.create_umap_plot(
        df=initial_df, 
        output_directory=output_directory,
        feature_columns=target_proteins, 
        color_column=CLASSIFIED_BY, 
        metadata_cols=[SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC GROUP'],
        n_neighbors=5,
        )
    
    # TCC vs Probability plot
    TCC_plot = grph.plot_tcc_vs_probability(initial_df, test_target_scores, output_directory)
    
    return TCC_plot, UMAP_plot

def print_configuration():
    """Print current configuration settings"""
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    print(f"Target Class: {TARGET_CLASS}")
    print(f"Classification Column: {CLASSIFIED_BY}")
    print(f"Data Folder: {PROCESSED_DATA_FOLDER}")
    print(f"Split Size: {SPLIT_SIZE}")
    print(f"High Confidence Threshold: {HIGH_CONFIDENCE_THRESHOLD}")
    print(f"ElasticNet Parameters: L1_ratio={ELNET_L1_RATIO}, C={ELNET_C_VALUE}")
    print(f"Cross-validation: {ELNET_N_SPLITS} splits, {ELNET_N_REPEATS} repeats")
    print(f"Nested CV: {NESTED_CV_RANDOM_STATE_TRIES} tries, {NESTED_CV_N_SPLITS} splits")
    print("=" * 80)


###############################
### Main Execution Function ###
###############################

def main():
    """Main execution function"""
    print("#" * 80)
    print("★Entity Classifier Generator★")
    print("#" * 80)

    # Print configuration
    print_configuration()
    
    # Setup paths and import modules
    project_root = setup_paths()
    print(f'Expected output directory: {output_dir}')
    prep, fs, mf, grph = import_custom_modules()
    
    # Load data
    held_out_df, train_df, target_proteins = load_data(prep)
    
    # Preprocess data
    training_df, held_out_df, scaler = preprocess_data(
        held_out_df, train_df
    )

    # Binary labelling of dataframes 
    target_training_df, target_ho_df = binary_labelling(
        training_df, held_out_df, fs
    )
  
    # Model fitting
    model_results = model_fitting(target_training_df, target_ho_df, target_proteins, output_dir,fs, mf)


    if model_results[0] is not None:
        print("=" * 80)
        print(f"CLASSIFIER WORKFLOW COMPLETED SUCCESSFULLY FOR {TARGET_CLASS}!")
        print("=" * 80)
        print(f"Selected {len(target_proteins)} protein features")
        print("Check the 'data/data_output' directory for exported results")
    else:
        print("=" * 80)
        print("CLASSIFIER WORKFLOW COMPLETED WITH ERRORS")
        print("=" * 80)
    
    return model_results




if __name__ == "__main__":
    original_stdout = sys.stdout
    try:
        sys.stdout = tee_output
        results = main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = original_stdout
        tee_output.close()
        print(f"Process output saved to: {output_filename}")
