# Entity Classifier Generator

A machine learning pipeline for entity classification using protein quantification data. This project implements a complete workflow from data preprocessing to model evaluation using logistic regression with elastic net regularization.

## 🔬 Overview

This classifier processes proteomic data to identify biomarkers and build predictive models for entity classification. The pipeline includes:

- Data preprocessing and imputation
- Feature selection using ElasticNet
- Cross-validation and hyperparameter tuning
- Model training and evaluation
- Automated result export

## 📁 Project Structure

```
tumor_type_prediction/
├── scripts/
│   └── entity_classifier_script.py      # Main execution script
├── src/
│   └── data/
│       ├── entity_model_settings.py     # Configuration parameters
│       ├── preprocessing.py             # Data preprocessing functions
│       ├── feature_selection.py         # Feature selection methods
│       └── model_fit.py                 # Model training and evaluation
├── data/
│   └── data_output/                     # Generated results and exports
│       ├── data_frames                  # dataframes used in the process
│       ├── feature_selection            # Cross validation and selected features
│       ├── model_fit                    # Model as pickle and results on test set
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn openpyxl pathlib
```

### Configuration

Edit `src/data/entity_model_settings.py` to configure:

- `TARGET_CLASS`: Entity class to classify
- `CLASSIFIED_BY`: Column name for classification
- `SAMPLES_COLUMN`: Column name for sample identifiers
- File paths and data locations
- Model hyperparameters
- Cross-validation settings

### Running the Script

```bash
cd scripts/
python entity_classifier_script.py
```

## 📊 Input Data

The pipeline expects three main data files:

1. **Intensity Data**: Protein quantification intensities (XLSX, CSV, TSV format)
2. **Z-scores Data**: Normalized protein expression values (XLSX, CSV, TSV format)  
3. **Metadata**: Sample metadata with classification labels (Excel format)

## 🔧 Key Features

### Data Preprocessing
- Missing value imputation using normal distribution downshift
- Sample name cleaning and standardization  
- Binary encoding for peptide quantification data
- Train/test data splitting with stratification

### Feature Selection
- High-confidence protein filtering based on peptide evidence
- ElasticNet regularization for feature selection
- Cross-validation for hyperparameter optimization
- Statistical analysis of feature coefficients

### Model Training
- Logistic regression with Ridge regularization
- Nested cross-validation for robust evaluation
- Hyperparameter tuning with grid search
- Model performance metrics and evaluation

### Output Generation
- Model coefficients and feature importance
- Training and test probabilities
- Classification performance scores
- Comprehensive logging and error handling

## ⚙️ Configuration Parameters

Parameters in `entity_model_settings.py`:

# Configuration Parameters
(To be modified to something standard)
## Data File Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PROCESSED_DATA_FOLDER` | `'Test_datasets_PROdict/'` | Folder containing processed data files |
| `FOLDER_PATH` | `'/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/LE_PROdict/'` | Base path for data files |
| `METADATA_PATH` | `'/media/kusterlab/internal_projects/active/TOPAS/WP31/Playground/Retrospective_MTBs_Evaluation/'` | Path to metadata files |

## File Names

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PREPROCESSED_FP_INTENSITY` | `'input_quantifications_small.csv'` | Protein intensity quantification file |
| `PREPROCESSED_FP_Z_SCORES` | `'df_z_scores_small.csv'` | Z-scores data file |
| `METADATA_FILE` | `'METADATA_PAN_CANCER_Batch300.xlsx'` | Sample metadata file |

## Classification Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TARGET_CLASS` | `['BRCA']` | Target entity class for classification |
| `CLASSIFIED_BY` | `'code_oncotree'` | Column name for classification labels |
| `SAMPLES_COLUMN` | `'Sample name'` | Column name for sample identifiers |

## Data Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NOS_CASES` | `['CUPNOS', 'ADNOS', 'SARCNOS', 'SCCNOS', 'SOLIDNOS', 'RCSNOS', 'GCTNOS']` | NOS (Not Otherwise Specified) cases to remove |
| `OTHER_CASES` | `['missing']` | Additional classes to exclude from analysis |
| `SPLIT_SIZE` | `0.25` | Proportion of data for held-out test set |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.7` | Minimum threshold for high confidence proteins |

## Feature Selection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FEATURE_SELECTION_L1_RATIOS` | `[0.5, 0.7]` | L1 regularization ratios for ElasticNet grid search |
| `FEATURE_SELECTION_C_VALUES` | `[0.1, 1]` | Inverse regularization strength values for grid search |
| `GRID_SEARCH_N_SPLITS` | `4` | Number of CV folds for hyperparameter grid search |
| `ELNET_L1_RATIO` | `0.5` | L1 ratio for ElasticNet feature selection |
| `ELNET_C_VALUE` | `1` | Inverse regularization strength for ElasticNet |
| `ELNET_N_SPLITS` | `4` | Number of CV folds for ElasticNet |
| `ELNET_N_REPEATS` | `25` | Number of repeated CV runs for ElasticNet |
| `ELNET_N_JOBS` | `16` | Number of parallel jobs for ElasticNet |

## Model Fitting Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NESTED_CV_RANDOM_STATE_TRIES` | `5` | Number of random states for nested cross-validation |
| `NESTED_CV_N_SPLITS` | `3` | Number of folds for nested cross-validation |

## Imputation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMPUTATION_WIDTH` | `0.3` | Width parameter for normal distribution imputation |
| `IMPUTATION_DOWNSHIFT` | `1.8` | Downshift parameter for normal distribution imputation |
| `IMPUTATION_SEED` | `2` | Random seed for reproducible imputation |


## 📈 Output Files

The script generates several output files in `data/data_output/`:

- Model coefficients and weights
- Feature selection results  
- Cross-validation results
- Training and test predictions
- Performance metrics and scores

## 🔍 Monitoring and Logging

The script provides comprehensive logging:

- **Output Log**: All print statements saved to timestamped file
- **Warning Log**: Warnings and errors logged separately
- **Error Handling**: Graceful error handling with detailed messages


## 📝 Usage Example

1. Locate yourself in the root of the project directory
2. Activate your environment
3. From the command line run
```bash
python scripts/entity_classifier_script.py
```
4. Check for the output results in the folder ```tumor_type_prediction/data/data_output```
5. Check at the summary results of the pipeline in ```tumor_type_prediction/classifier_output_{date}.txt```

