#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_ANGS.py
python scripts/entity_classifier_script_BA.py  
python scripts/entity_classifier_script_CHS.py
python scripts/entity_classifier_script_DFSP.py
python scripts/entity_classifier_script_GINET.py
python scripts/entity_classifier_script_THYM.py

echo "All scripts finished at $(date)"