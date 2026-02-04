"""
Classifier Workflow Script
Command-line runnable version of the original Jupyter notebook
"""

############
# Imports ##
############
import os
import sys
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore
import warnings
import logging
import multiprocessing as mp

from datetime import datetime
from pathlib import Path

import prodict.preprocessing as prep
import prodict.feature_selection as fs
import prodict.model_fit as mf
import prodict.graphs as grph
import prodict.config as cfg

from sklearn.preprocessing import StandardScaler
import pickle

# Importing settings from YAML configuration for the model
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ProDICT
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "small_data_model_settings.yaml"
CONFIG_PATH = Path(os.environ.get("PRODICT_CONFIG", str(DEFAULT_CONFIG)))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = (PROJECT_ROOT / CONFIG_PATH).resolve()

cfg.load_config(CONFIG_PATH)
# lestrada@linux-cluster:~/projects/ProDICT/src/prodict$ PRODICT_CONFIG=../ProDICT/data/small_data_model_settings.yaml python -m prodict.entity_classifier_script_blueprint

# Derive output directory consistently
output_dir = PROJECT_ROOT / "data" / cfg.RUN_FOLDER_NAME
output_dir.mkdir(parents=True, exist_ok=True)
from prodict.config import *
###################
# Logging Setup ###
###################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = output_dir / f"classifier_log_{timestamp}.log"
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


output_filename = output_dir / f"classifier_output_{timestamp}.txt"
tee_output = TeeOutput(output_filename)

#############
# Functions #
#############


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


def load_data():
    """Load all required data files"""
    print("=" * 80)
    print("Loading data files...")
    print("=" * 80)
    print(f"Target class: {TARGET_CLASS}")
    print(f"Classification column: {CLASSIFIED_BY}")

    # Construct file paths using configuration variables
    intensity_path_file = FOLDER_PATH + PROCESSED_DATA_FOLDER + PREPROCESSED_FP_INTENSITY
    # z_scores_path_file = FOLDER_PATH + PROCESSED_DATA_FOLDER + PREPROCESSED_FP_Z_SCORES
    the_metadata_file = METADATA_PATH + METADATA_FILE

    print(f"Loading intensity data from: {intensity_path_file}")
    # print(f"Loading z-scores data from: {z_scores_path_file}")
    print(f"Loading metadata from: {the_metadata_file}")

    try:
        input_quantifications = prep.read_table_with_correct_sep(intensity_path_file)
        # df_z_scores = prep.read_table_with_correct_sep(z_scores_path_file)
        input_metadata = pd.read_excel(
            the_metadata_file,
            usecols=['Sample name', 'code_oncotree', 'Tumor cell content', 'TCC_Bioinfo', 'TCC GROUP'],
            dtype={'Sample name': 'string', 'code_oncotree': 'string', 'Tumor cell content': 'float64', 'TCC_Bioinfo': 'float64', 'TCC GROUP': 'string'},
            na_values=['', 'NA', 'NaN', 'nan', 'N/A', 'n/a', 'None', 'TBD', 'notavailable', 'missing'])

        print("Data files loaded successfully.")
        print(f"Quantifications shape: {input_quantifications.shape}")
        # print(f"Z-scores shape: {df_z_scores.shape}")
        print(f"Metadata shape: {input_metadata.shape}")

        return input_quantifications, input_metadata

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please check that all data files exist in the specified paths:")
        print(f"  - {intensity_path_file}")
        # print(f"  - {z_scores_path_file}")
        print(f"  - {the_metadata_file}")
        sys.exit(1)


def preprocess_data(input_quantifications, input_metadata):
    """Preprocess all data"""
    print("=" * 80)
    print("Preprocessing data...")
    print("=" * 80)
    # Protein quantification intensities post-processing

    input_quantifications = input_quantifications.set_index(input_quantifications.columns[0])  # noqa: E501
    peptides_quant_info = prep.post_process_meta_intensities(
        input_quantifications.iloc[:, int(input_quantifications.shape[1] / 2):].T
    )
    proteins_quant = input_quantifications.iloc[:, :int(input_quantifications.shape[1] / 2)].T
    print(f"***proteins quantifications columns: {proteins_quant.iloc[:,:10].columns.tolist()}")

    # Cleaning sample names
    prot_quant_imputed = proteins_quant.copy()  # Placeholder for imputation step
    prot_quant_imputed.reset_index(inplace=True)
    prot_quant_imputed.rename(columns={'index': SAMPLES_COLUMN}, inplace=True)
    prot_quant_imputed[SAMPLES_COLUMN] = prot_quant_imputed[SAMPLES_COLUMN].str.replace('pat_', '').str.strip()

    # Dataset with protein intensities and metadata
    input_metadata['TCC'] = input_metadata['TCC_Bioinfo'].fillna(input_metadata['Tumor cell content'])
    samples_metadata = input_metadata[[SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC', 'TCC GROUP']]
    samples_metadata[SAMPLES_COLUMN] = samples_metadata[SAMPLES_COLUMN].str.strip()
    initial_df = samples_metadata.merge(prot_quant_imputed, on=SAMPLES_COLUMN, how='left')

    # Peptides quantification to binary dataset
    peptides_df_binary = pd.DataFrame(
        np.where(peptides_quant_info > 1, 1, 0),
        index=peptides_quant_info.index,
        columns=peptides_quant_info.columns
    )
    peptides_df_binary.reset_index(inplace=True)
    peptides_df_binary.replace('Identification metadata ', '', regex=True, inplace=True)
    peptides_df_binary['index'] = peptides_df_binary['index'].str.strip()
    peptides_df_binary = samples_metadata.merge(peptides_df_binary, left_on=SAMPLES_COLUMN, right_on='index')
    peptides_df_binary.drop('index', axis=1, inplace=True)

    print("Peptides binary dataframe shape:", peptides_df_binary.shape)

    return initial_df, peptides_df_binary


def split_data(initial_df, output_directory, export_train_split):
    """Split data into training and held-out sets.
        Train is z-score normalized and imputed. Test is normalized with train parameters and imputed.

    Returns:
        scaled_train: z-score normalized and imputed training set
        scaled_hold_out: z-score normalized and imputed held-out set
    """
    print("=" * 80)
    print("Splitting data...")
    print("=" * 80)
    nos_cases = initial_df[initial_df[CLASSIFIED_BY].str.endswith('NOS', na=False)][CLASSIFIED_BY].unique().tolist()
    cases_to_remove = nos_cases + OTHER_CASES
    print(f"Removing undefined cases: {cases_to_remove}")

    # Removing samples not part of the Oncotree classification
    ml_initial_df = (
        initial_df
        .pipe(prep.remove_class, cases_to_remove, CLASSIFIED_BY, output_directory)
        .pipe(prep.remove_class, ['very low', 'notdefined'], 'TCC GROUP', output_directory)
    )

    # Splitting dataset into training and held-out sets
    training_df, held_out_df = prep.data_split(
        ml_initial_df,
        output_directory=output_directory,
        split_size=SPLIT_SIZE,
        classified_by=CLASSIFIED_BY,
        export=export_train_split,
    )

    print("=" * 80)
    print("Preprocessing data, normalizing and imputing Train and Test set...")
    print("=" * 80)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(training_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1, errors='ignore'))
    scaled_hold_out = scaler.transform(held_out_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1, errors='ignore'))

    with open(f"{output_dir}/{TARGET_CLASS_NAME}_normalization_parameters.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Imputing train and test
    train_scaled_imputed = prep.impute_normal_down_shift_distribution(
        pd.DataFrame(scaled_train),
        width=IMPUTATION_WIDTH,
        downshift=IMPUTATION_DOWNSHIFT,
        seed=IMPUTATION_SEED
    )
    train_scaled_imputed.columns = training_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1, errors='ignore').columns
    training_df_reset_index = training_df[['Sample name', 'code_oncotree']].reset_index(drop=True)
    scaled_train = pd.concat([training_df_reset_index, train_scaled_imputed], axis=1)

    ho_scaled_imputed = prep.impute_normal_down_shift_distribution(
        pd.DataFrame(scaled_hold_out),
        width=IMPUTATION_WIDTH,
        downshift=IMPUTATION_DOWNSHIFT,
        seed=IMPUTATION_SEED
    )
    ho_scaled_imputed.columns = held_out_df.drop(['Sample name', 'code_oncotree', 'TCC', 'TCC GROUP'], axis=1, errors='ignore').columns
    scaled_hold_out = pd.concat([held_out_df[['Sample name', 'code_oncotree']].reset_index(drop=True), ho_scaled_imputed], axis=1)

    print("Train dataframe shape:", training_df.shape)
    print("Train Normalized dataframe shape:", scaled_train.shape)
    print("Test dataframe shape:", held_out_df.shape)
    print("Test Normalized dataframe shape:", scaled_hold_out.shape)
    print(f"Samples match between Z-score and intesntity dataset: {set(training_df['Sample name']) == set(scaled_train['Sample name'])}")
    print(f"Training set size: {training_df.shape}")
    print(f"Held-out set size: {held_out_df.shape}")

    return training_df, held_out_df, scaled_train, scaled_hold_out


def class_specific_workflow(training_df, held_out_df, scaled_train, scaled_hold_out, peptides_df_binary, output_directory, tumor_type_name=TARGET_CLASS_NAME):
    """Execute class-specific workflow for specified classification"""
    print("=" * 80)
    print(f"Starting class-specific workflow for {TARGET_CLASS}...")
    print("=" * 80)
    # Obtaining high confidence proteins by peptides
    target_proteins_by_peptides = fs.get_high_confidence_proteins(
        peptides_df_binary, TARGET_CLASS, CLASSIFIED_BY, threshold=HIGH_CONFIDENCE_THRESHOLD
    )

    # Binary labeling for specific class classification
    target_training_df =        fs.binary_labeling(training_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    target_ho_df =              fs.binary_labeling(held_out_df, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    target_z_scores_train_df =  fs.binary_labeling(scaled_train, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)
    target_z_scores_held_out_df =  fs.binary_labeling(scaled_hold_out, classified_by=CLASSIFIED_BY, true_class=TARGET_CLASS)

    # 1st Filter - Filtering training and held-out dataframes by proteins with peptides
    target_training_df = target_training_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + target_proteins_by_peptides)

    # 2nd Filter - Filtering taining and held-out dataframes by mann whitney U significant test
    effect_size_for_class = fs.calculate_mann_whitney(
        target_training_df, output_directory, tumor_type_name, exclude_cols=['code_oncotree', 'TCC', 'Classifier'])
    significant_features_mwu = list(effect_size_for_class[(effect_size_for_class['p_value_adj'] < 0.05) & (effect_size_for_class['cliffs_delta'].abs()> 0.147)]['feature'])

    target_training_df = target_training_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + significant_features_mwu)
    target_z_scores_train_df = target_z_scores_train_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + significant_features_mwu)

    #target_z_scores_train_df = target_z_scores_train_df.filter(items=[SAMPLES_COLUMN, CLASSIFIED_BY, 'Classifier'] + target_proteins_by_peptides)

    print(f"Filtered training set shape: {target_training_df.shape}")
    print(f"Filtered held-out set shape: {target_ho_df.shape}")
    print(f"Filtered z-scores training set shape: {target_z_scores_train_df.shape}")
    print('*' * 80)
    # print(f"{len(significant_features_mwu)} significant proteins (p<0.01 & Cliff's d > 0.15)")
    # print(f"{significant_features_mwu[:10]}")

    return target_training_df, target_ho_df, target_z_scores_train_df, target_z_scores_held_out_df


def feature_selection(target_z_scores_train_df, output_directory):
    """Perform feature selection using ElasticNet"""
    print("="*80)
    print("Starting feature selection...")
    print("="*80)
    print(f"Using L1 ratios: {FEATURE_SELECTION_L1_RATIOS}")
    print(f"Using C values: {FEATURE_SELECTION_C_VALUES}")

    # Hyperparameters for ElasticNet
    print("-" * 80)
    print("Defining hyperparameters for ElasticNet...")
    print(f"Number of proteins used:{target_z_scores_train_df.shape[1]}")

    try:
        target_cv_results, target_best_params, target_best_score, target_grid_search_obj = fs.hparameter_grid_search(
            target_z_scores_train_df, GRID_SEARCH_N_SPLITS, FEATURE_SELECTION_L1_RATIOS, FEATURE_SELECTION_C_VALUES, classified_by=CLASSIFIED_BY
        )

    except Exception as e:
        print(f"Warning: Hyperparameter search failed: {e}")
        print("Using configured default parameters...")
        target_best_params = {'l1_ratio': ELNET_L1_RATIO, 'C': ELNET_C_VALUE}

    # Feature Selection by ElasticNet Cross-Validation
    print("-" * 80)
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


def model_fitting(target_training_df, target_ho_df, target_proteins, output_directory):
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


def generate_graphs(training_df, held_out_df, test_target_scores, target_proteins, output_directory):
    """Generate and save graphs for results exploration"""
    print("="*80)
    print("Generating graphs...")
    print("="*80)

    initial_df = pd.concat([training_df, held_out_df], ignore_index=True, axis=0)
    prot_quant_imputed = prep.impute_normal_down_shift_distribution(
        initial_df.drop([SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC', 'TCC GROUP', 'Classifier'], axis=1, errors='ignore'),
        width=IMPUTATION_WIDTH,
        downshift=IMPUTATION_DOWNSHIFT,
        seed=IMPUTATION_SEED
    )
    initial_df_imputed = pd.concat(
        [initial_df[[SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC']].reset_index(drop=True), prot_quant_imputed.reset_index(drop=True)],
        axis=1
    )

    # UMAP plot
    UMAP_plot = grph.create_umap_plot(
        df=initial_df_imputed,
        output_directory=output_directory,
        feature_columns=target_proteins,
        color_column=CLASSIFIED_BY,
        metadata_cols=[SAMPLES_COLUMN, CLASSIFIED_BY, 'TCC'],
        n_neighbors=5,
        )

    # TCC vs Probability plot
    # TCC_plot = grph.plot_tcc_vs_probability(initial_df_imputed, test_target_scores, output_directory)

    return UMAP_plot


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


###########################
# Main Execution Function #
###########################


def main():
    """Main execution function"""
    print("#" * 80)
    print("★Entity Classifier Generator★")
    print("#" * 80)

    # Print configuration
    print_configuration()

    # Setup paths and import modules
    print(f'Expected output directory: {output_dir}')
    # prep, fs, mf, grph = import_custom_modules()

    # Load data
    input_quantifications, input_metadata = load_data()

    # Preprocess data
    initial_df, peptides_df_binary = preprocess_data(
        input_quantifications, input_metadata
    )

    # Split data
    training_df, held_out_df, z_scores_train_df, z_scores_held_out = split_data(
        initial_df, output_dir, export_train_split=False
    )

    # Class-specific workflow
    target_training_df, target_ho_df, target_z_scores_train_df, target_z_scores_held_out_df = class_specific_workflow(
        training_df, held_out_df, z_scores_train_df, z_scores_held_out, peptides_df_binary, output_dir, tumor_type_name=TARGET_CLASS_NAME
        # peptides_df_binary might introduce data leakage
        # peptideds_df_binary was calculates with all samples and not just training samples
    )

    # Feature selection
    target_proteins = feature_selection(target_z_scores_train_df, output_dir)

    # Model fitting
    model_results = model_fitting(target_z_scores_train_df, target_z_scores_held_out_df, target_proteins, output_dir)

    # Generate graphs
    generate_graphs(training_df, held_out_df, model_results[4], target_proteins, output_dir)

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