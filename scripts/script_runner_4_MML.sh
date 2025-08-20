#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_ARMS.py
python scripts/entity_classifier_script_ERMS.py  
python scripts/entity_classifier_script_OS.py
python scripts/entity_classifier_script_ES.py
python scripts/entity_classifier_script_ULMS.py
python scripts/entity_classifier_script_ACC.py


echo "All scripts finished at $(date)"