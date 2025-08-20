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


current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
module_path = project_root / "src" / "data"
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
    
from entity_model_settings_ES import *

project_root = os.path.abspath(os.getcwd())
output_dir = os.path.join(project_root, 'data', run_folder_name)
os.makedirs(output_dir, exist_ok=True)

#####################
### Logging Setup ###
#####################

log_filename = f"classifier_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),

    ]
)


warnings.filterwarnings("default") 
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(
    f"{category.__name__}: {message} (in {filename}:{lineno})"
)


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

def load_data(project_root, prep):
    """Load all required data files"""
    print("="*80)
    print("Loading data files...")
    print("="*80)
    print(f"Target class: {TARGET_CLASS}")
    print(f"Classification column: {CLASSIFIED_BY}")
    
    # Construct file paths using configuration variables
    intensity_path_file = FOLDER_PATH + PROCESSED_DATA_FOLDER + PREPROCESSED_FP_INTENSITY
    z_scores_path_file = FOLDER_PATH + PROCESSED_DATA_FOLDER + PREPROCESSED_FP_Z_SCORES
    the_metadata_file = METADATA_PATH + METADATA_FILE
    
    print(f"Loading intensity data from: {intensity_path_file}")
    print(f"Loading z-scores data from: {z_scores_path_file}")
    print(f"Loading metadata from: {the_metadata_file}")
    
    try:
        input_quantifications = prep.read_table_with_correct_sep(intensity_path_file)
        df_z_scores = prep.read_table_with_correct_sep(z_scores_path_file)
        input_metadata = pd.read_excel(the_metadata_file,
                                        usecols=['Sample name', 'code_oncotree', 'Tumor cell content', 'TCC_Bioinfo', 'TCC GROUP'],
                                        dtype={'Sample name': 'string', 'code_oncotree': 'string', 'Tumor cell content': 'float64', 'TCC_Bioinfo': 'float64', 'TCC GROUP': 'string'},
                                        na_values=['', 'NA', 'NaN', 'nan', 'N/A', 'n/a', 'None', 'TBD', 'notavailable'])

        print("Data files loaded successfully.")
        print(f"Quantifications shape: {input_quantifications.shape}")
        print(f"Z-scores shape: {df_z_scores.shape}")
        print(f"Metadata shape: {input_metadata.shape}")

        return input_quantifications, df_z_scores, input_metadata

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please check that all data files exist in the specified paths:")
        print(f"  - {intensity_path_file}")
        print(f"  - {z_scores_path_file}")
        print(f"  - {the_metadata_file}")
        sys.exit(1)

def preprocess_data(input_quantifications, df_z_scores, input_metadata,prep):
    """Preprocess all data"""
    print("="*80)
    print("Preprocessing data...")
    print("="*80)
    # Protein quantification intensities post-processing


    input_quantifications = input_quantifications.set_index(input_quantifications.columns[0])
    peptides_quant_info = prep.post_process_meta_intensities(
        input_quantifications.iloc[:, int(input_quantifications.shape[1]/2):].T
    )
    proteins_quant = input_quantifications.iloc[:, :int(input_quantifications.shape[1]/2)].T
    print(f"***proteins quantifications columns: {proteins_quant.iloc[:,:30].columns.tolist()}")
    # Imputation with configurable parameters
    prot_quant_imputed = prep.impute_normal_down_shift_distribution(
        proteins_quant, 
        width=IMPUTATION_WIDTH, 
        downshift=IMPUTATION_DOWNSHIFT, 
        seed=IMPUTATION_SEED
    )
    na_columns = prot_quant_imputed.isna().any()
    na_columns_true = na_columns[na_columns].index.tolist()
    print("Proteins with empty values:", na_columns_true)

    print(f"***Input after imputation columns: {prot_quant_imputed.iloc[:,:30].columns.tolist()}")
    # Cleaning sample names
    prot_quant_imputed.reset_index(inplace=True)
    prot_quant_imputed.rename(columns={'index': SAMPLES_COLUMN}, inplace=True)
    prot_quant_imputed[SAMPLES_COLUMN] = prot_quant_imputed[SAMPLES_COLUMN].str.replace('pat_', '')
    
    # Dataset with protein intensities and metadata
    input_metadata['TCC'] = input_metadata['TCC_Bioinfo'].fillna(input_metadata['Tumor cell content'])
    samples_metadata = input_metadata[[SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC', 'TCC GROUP']]
    
    initial_df = samples_metadata.merge(prot_quant_imputed, left_on=SAMPLES_COLUMN, right_on=SAMPLES_COLUMN)
    
    # Peptides quantification to binary dataset
    peptides_df_binary = pd.DataFrame(
        np.where(peptides_quant_info > 1, 1, 0),
        index=peptides_quant_info.index,
        columns=peptides_quant_info.columns  
    )
    peptides_df_binary.reset_index(inplace=True)
    peptides_df_binary.replace('Identification metadata ', '', regex=True, inplace=True)
    peptides_df_binary = samples_metadata.merge(peptides_df_binary, left_on=SAMPLES_COLUMN, right_on='index')
    peptides_df_binary.drop('index', axis=1, inplace=True)
    
    print("Peptides binary dataframe shape:", peptides_df_binary.shape)
    
    # Process Z-scores
    z_scores_df = df_z_scores.transpose(copy=True) 
    print("Z-scores dataframe shape before processing:", z_scores_df.shape)
    z_scores_df = z_scores_df.reset_index()
    z_scores_df = z_scores_df.replace('zscore_','', regex=True) 
    z_scores_df.rename(columns = z_scores_df.iloc[0], inplace=True)
    z_scores_df.drop(axis=0, index=0, inplace=True)
    z_scores_df['Gene names'] = z_scores_df.iloc[:,0].str.replace('pat_', '')
    z_scores_df = z_scores_df.set_index('Gene names') 
    print("Z-scores dataframe shape after processing:", z_scores_df.shape)
    #print("Z-scores columns:", z_scores_df.columns.tolist())
    z_scores_imputed = prep.impute_normal_down_shift_distribution(
        z_scores_df,
        width=IMPUTATION_WIDTH, 
        downshift=IMPUTATION_DOWNSHIFT, 
        seed=IMPUTATION_SEED
    )
    z_scores_imputed.reset_index(inplace=True)
    z_scores_imputed.rename(columns={'Gene names': SAMPLES_COLUMN}, inplace=True)

    z_scores_initial_df = samples_metadata.merge(z_scores_imputed, left_on=SAMPLES_COLUMN, right_on=SAMPLES_COLUMN)
    
    print("Z-scores initial dataframe shape:", z_scores_initial_df.shape)
    
    return initial_df, peptides_df_binary, z_scores_initial_df

def split_data(initial_df, z_scores_initial_df, output_directory, prep):
    """Split data into training and held-out sets"""
    print("="*80)
    print("Splitting data...")
    print("="*80)
    nos_cases = initial_df[initial_df[CLASSIFIED_BY].str.endswith('NOS', na=False)][CLASSIFIED_BY].unique().tolist()
    cases_to_remove = nos_cases + OTHER_CASES
    print(f"Removing undefined cases: {cases_to_remove}")

    # Removing samples not part of the Oncotree classification
    ml_initial_df = (
        initial_df
        .pipe(prep.remove_class, cases_to_remove, CLASSIFIED_BY, output_directory)
        .pipe(prep.remove_class, ['very low', 'missing'], 'TCC GROUP', output_directory)
        .loc[lambda df: df['TCC GROUP'].notna()]
    )


    # Splitting dataset into training and held-out sets
    training_df, held_out_df = prep.data_split(
        ml_initial_df,
        output_directory=output_directory, 
        split_size=SPLIT_SIZE, 
        classified_by=CLASSIFIED_BY, 
        export=False,
    )
    
    # Z_scores dataset
    z_scores_train_df = z_scores_initial_df[z_scores_initial_df['Sample name'].isin(training_df['Sample name'])]

    print(f"Samples match between Z-score and intesntity dataset: {set(training_df['Sample name']) == set(z_scores_train_df['Sample name'])}")
    
    print(f"Training set size: {training_df.shape}")
    print(f"Held-out set size: {held_out_df.shape}")
    print(f"Z-scores training set size: {z_scores_train_df.shape}")
    
    return training_df, held_out_df, z_scores_train_df

def class_specific_workflow(training_df, held_out_df, z_scores_train_df, peptides_df_binary, fs, mf):
    """Execute class-specific workflow for specified classification"""
    print("="*80)
    print(f"Starting class-specific workflow for {TARGET_CLASS}...")
    print("="*80)
    # Obtaining high confidence proteins by peptides
    target_proteins_by_peptides = fs.get_high_confidence_proteins(
        peptides_df_binary, TARGET_CLASS, CLASSIFIED_BY, threshold=HIGH_CONFIDENCE_THRESHOLD
    )
    
    # Binary labeling for specific class classification 
    target_training_df =        fs.binary_labeling(training_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    target_ho_df =              fs.binary_labeling(held_out_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    target_z_scores_train_df =  fs.binary_labeling(z_scores_train_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    
    # 1st Filter - Filtering training and held-out dataframes by proteins with peptides
    target_training_df = target_training_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + target_proteins_by_peptides)
    target_ho_df = target_ho_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + target_proteins_by_peptides)
    target_z_scores_train_df = target_z_scores_train_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + target_proteins_by_peptides)

    print(f"Filtered training set shape: {target_training_df.shape}")
    print(f"Filtered held-out set shape: {target_ho_df.shape}")
    print(f"Filtered z-scores training set shape: {target_z_scores_train_df.shape}")

    return target_training_df, target_ho_df, target_z_scores_train_df

def feature_selection(target_z_scores_train_df, output_directory, fs):
    """Perform feature selection using ElasticNet"""
    print("="*80)
    print("Starting feature selection...")
    print("="*80)
    print(f"Using L1 ratios: {FEATURE_SELECTION_L1_RATIOS}")
    print(f"Using C values: {FEATURE_SELECTION_C_VALUES}")
    
    # Hyperparameters for ElasticNet
    print("-"*80)
    print("Defining hyperparameters for ElasticNet...")

    try:
        target_cv_results, target_best_params, target_best_score, target_grid_search_obj = fs.hparameter_grid_search(
            target_z_scores_train_df, GRID_SEARCH_N_SPLITS, FEATURE_SELECTION_L1_RATIOS, FEATURE_SELECTION_C_VALUES, classified_by=CLASSIFIED_BY
        )
        
    except Exception as e:
        print(f"Warning: Hyperparameter search failed: {e}")
        print("Using configured default parameters...")
        target_best_params = {'l1_ratio': ELNET_L1_RATIO, 'C': ELNET_C_VALUE}

    # Feature Selection by ElasticNet Cross-Validation
    print("-"*80)
    print("Selecting features...")

    try:
        class_name = "_".join(TARGET_CLASS)
        target_cross_val_coeffs = fs.elnet_wrapper(
            target_z_scores_train_df, 
            classified_by=CLASSIFIED_BY, 
            tumor_type_name=f'{class_name}_features', 
            l1_ratio=target_best_params.get('l1_ratio'), 
            C=target_best_params.get('C'), 
            output_directory = output_directory,
            n_splits=ELNET_N_SPLITS, 
            n_repeats=ELNET_N_REPEATS, 
            n_jobs=ELNET_N_JOBS, 
            export=True
        )
        
        target_stats, target_proteins = fs.statistic_from_coefficients(target_cross_val_coeffs, TARGET_CLASS, output_directory)
        
    except Exception as e:
        print(f"Warning: Feature selection failed: {e}")
            
    print(f"Selected {len(target_proteins)} protein features")
    return target_proteins
    
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
    input_quantifications, df_z_scores, input_metadata = load_data(project_root, prep)
    
    # Preprocess data
    initial_df, peptides_df_binary, z_scores_initial_df = preprocess_data(
        input_quantifications, df_z_scores, input_metadata, prep
    )
    
    # Split data
    training_df, held_out_df, z_scores_train_df = split_data(
        initial_df, z_scores_initial_df, output_dir, prep
    )
    
    # Class-specific workflow
    target_training_df, target_ho_df, target_z_scores_train_df = class_specific_workflow(
        training_df, held_out_df, z_scores_train_df, peptides_df_binary, fs, mf
    )
    
    # Feature selection
    target_proteins = feature_selection(target_z_scores_train_df, output_dir, fs)
    
    # Model fitting
    model_results = model_fitting(target_training_df, target_ho_df, target_proteins, output_dir,fs, mf)

    # Generate graphs
    generate_graphs(initial_df, model_results[4], target_proteins, output_dir ,grph)

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
    except ChildProcessError as e:
        logging.error(f"Multiprocessing error: {e}")
        print(f"Multiprocessing error logged to {log_filename}")
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
        print(f"Warnings and errors logged to: {log_filename}")