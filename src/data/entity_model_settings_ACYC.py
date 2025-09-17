# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================
from datetime import datetime


# Data file parameters
PROCESSED_DATA_FOLDER = '2025.08.06_CJ_paper_final/'
FOLDER_PATH = '/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/Retrospective_study/'
METADATA_PATH = '/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/LE_PROdict/paper_data/'

# File names
PREPROCESSED_FP_INTENSITY = 'preprocessed_fp.csv'
PREPROCESSED_FP_Z_SCORES = 'full_proteome_measures_z.tsv'
METADATA_FILE = 'METADATA_PANCANCER_PAPER_final.xlsx'

# Classification parameters
TARGET_CLASS = ['ACYC']  # Change this for different tumor types (e.g., ['BRCA'], ['LUAD'], etc.)
CLASSIFIED_BY = 'code_oncotree'  # Column name for classification - default is 'code_oncotree'
SAMPLES_COLUMN = 'Sample name'  # Column name for sample identifiers - default is 'Sample name'
CELL_CONTENT_COLUMN = 'TCC_Bioinfo'  # Column name for tumor cell content - default is 'TCC_Bioinfo'

# Data processing parameters
NOS_CASES = ['CUPNOS', 'ADNOS', 'SARCNOS', 'SCCNOS', 'SOLIDNOS', 'RCSNOS', 'GCTNOS']
OTHER_CASES = ['missing'] # Additional classes to remove - Default is 'missing'
SPLIT_SIZE = 0.30  # Proportion for held-out set
HIGH_CONFIDENCE_THRESHOLD = 0.7  # Threshold for high confidence proteins

# Feature selection parameters
FEATURE_SELECTION_L1_RATIOS = [0.3,0.5]  # L1 ratios to test for ElasticNet
FEATURE_SELECTION_C_VALUES = [2, 3.5, 5]      # C values to test for ElasticNet
GRID_SEARCH_N_SPLITS = 3           # Number of splits for grid search

ELNET_L1_RATIO = 0.5                     # fallback value L1 ratio for ElasticNet hyperparameter selection
ELNET_C_VALUE = 1                        # fallback value C value for ElasticNet hyperparameter selection
ELNET_N_SPLITS = 3                       # Number of splits for cross-validation
ELNET_N_REPEATS = 67                     # Number of repeats for cross-validation
ELNET_N_JOBS = 12                        # Number of parallel jobs

# Model fitting parameters
NESTED_CV_RANDOM_STATE_TRIES = 10    # Number of random state tries for nested CV
NESTED_CV_N_SPLITS = 3              # Number of splits for nested CV

# Imputation parameters
IMPUTATION_WIDTH = 0.3              # Width parameter for normal distribution imputation
IMPUTATION_DOWNSHIFT = 1.8          # Downshift parameter for normal distribution imputation
IMPUTATION_SEED = 2                 # Random seed for imputation


# Output directory (DO NOT MODIFY)
timestamp = datetime.now().strftime('%y%m%d')
target_class_name = "_".join(TARGET_CLASS) if isinstance(TARGET_CLASS, list) else str(TARGET_CLASS)
run_folder_name = f"{target_class_name}_{timestamp}_results"
# =============================================================================
# END CONFIGURATION 
# =============================================================================
