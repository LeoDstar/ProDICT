#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_BRCA.py
python scripts/entity_classifier_script_ASPS.py



echo "All scripts finished at $(date)"
