from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROCESSED_DATA_FOLDER: str | None = None
FOLDER_PATH: str | None = None
METADATA_PATH: str | None = None

# filenames
PREPROCESSED_FP_INTENSITY: str | None = None
PREPROCESSED_FP_Z_SCORES: str | None = None
METADATA_FILE: str | None = None

# classification
TARGET_CLASS: list[str] | None = None
CLASSIFIED_BY: str | None = None
SAMPLES_COLUMN: str | None = None
CELL_CONTENT_COLUMN: str | None = None

# processing
NOS_CASES: list[str] | None = None
OTHER_CASES: list[str] | None = None
SPLIT_SIZE: float | None = None
HIGH_CONFIDENCE_THRESHOLD: float | None = None

# feature selection
FEATURE_SELECTION_L1_RATIOS: list[float] | None = None
FEATURE_SELECTION_C_VALUES: list[int] | None = None
GRID_SEARCH_N_SPLITS: int | None = None

ELNET_L1_RATIO: float | None = None
ELNET_C_VALUE: int | None = None
ELNET_N_SPLITS: int | None = None
ELNET_N_REPEATS: int | None = None
ELNET_N_JOBS: int | None = None

# model fitting
NESTED_CV_RANDOM_STATE_TRIES: int | None = None
NESTED_CV_N_SPLITS: int | None = None

# imputation
IMPUTATION_WIDTH: float | None = None
IMPUTATION_DOWNSHIFT: float | None = None
IMPUTATION_SEED: int | None = None

# derived (computed)
TARGET_CLASS_NAME: str | None = None
RUN_FOLDER_NAME: str | None = None


def load_config(config_path: Path) -> None:
    """
    Load YAML config and populate module-level variables to preserve
    the existing 'globals-like' pattern in main.py.
    """
    global TARGET_CLASS, CLASSIFIED_BY, FOLDER_PATH, PROCESSED_DATA_FOLDER
    global PREPROCESSED_FP_INTENSITY, PREPROCESSED_FP_Z_SCORES
    global METADATA_PATH, METADATA_FILE, RUN_FOLDER_NAME, TARGET_CLASS_NAME
    global SAMPLES_COLUMN, CELL_CONTENT_COLUMN
    global NOS_CASES, OTHER_CASES, SPLIT_SIZE, HIGH_CONFIDENCE_THRESHOLD
    global FEATURE_SELECTION_L1_RATIOS, FEATURE_SELECTION_C_VALUES, GRID_SEARCH_N_SPLITS
    global ELNET_L1_RATIO, ELNET_C_VALUE, ELNET_N_SPLITS, ELNET_N_REPEATS, ELNET_N_JOBS
    global NESTED_CV_RANDOM_STATE_TRIES, NESTED_CV_N_SPLITS
    global IMPUTATION_WIDTH, IMPUTATION_DOWNSHIFT, IMPUTATION_SEED

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())

    # Map YAML keys (lowercase) -> your legacy uppercase variables
    TARGET_CLASS = raw["TARGET_CLASS"]
    CLASSIFIED_BY = raw["CLASSIFIED_BY"]
    FOLDER_PATH = raw["FOLDER_PATH"]
    PROCESSED_DATA_FOLDER = raw["PROCESSED_DATA_FOLDER"]
    PREPROCESSED_FP_INTENSITY = raw["PREPROCESSED_FP_INTENSITY"]
    PREPROCESSED_FP_Z_SCORES = raw["PREPROCESSED_FP_Z_SCORES"]
    METADATA_PATH = raw["METADATA_PATH"]
    METADATA_FILE = raw["METADATA_FILE"]

    SAMPLES_COLUMN = raw["SAMPLES_COLUMN"]
    CELL_CONTENT_COLUMN = raw["CELL_CONTENT_COLUMN"]

    NOS_CASES = raw["NOS_CASES"]
    OTHER_CASES = raw["OTHER_CASES"]
    SPLIT_SIZE = raw["SPLIT_SIZE"]
    HIGH_CONFIDENCE_THRESHOLD = raw["HIGH_CONFIDENCE_THRESHOLD"]

    # feature selection
    FEATURE_SELECTION_L1_RATIOS = raw["FEATURE_SELECTION_L1_RATIOS"]
    FEATURE_SELECTION_C_VALUES = raw["FEATURE_SELECTION_C_VALUES"]
    GRID_SEARCH_N_SPLITS = raw["GRID_SEARCH_N_SPLITS"]

    ELNET_L1_RATIO = raw["ELNET_L1_RATIO"]
    ELNET_C_VALUE = raw["ELNET_C_VALUE"]
    ELNET_N_SPLITS = raw["ELNET_N_SPLITS"]
    ELNET_N_REPEATS = raw["ELNET_N_REPEATS"]
    ELNET_N_JOBS = raw["ELNET_N_JOBS"]

    # model fitting
    NESTED_CV_RANDOM_STATE_TRIES = raw["NESTED_CV_RANDOM_STATE_TRIES"]
    NESTED_CV_N_SPLITS = raw["NESTED_CV_N_SPLITS"]

    # imputation
    IMPUTATION_WIDTH = raw["IMPUTATION_WIDTH"]
    IMPUTATION_DOWNSHIFT = raw["IMPUTATION_DOWNSHIFT"]
    IMPUTATION_SEED = raw["IMPUTATION_SEED"]

    # derived (computed)
    TARGET_CLASS_NAME = "_".join(TARGET_CLASS)
    RUN_FOLDER_NAME = f"{TARGET_CLASS_NAME}_{datetime.now():%y%m%d}_results"