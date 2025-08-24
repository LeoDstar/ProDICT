#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_BRCA.py
python scripts/entity_classifier_script_GINET.py
python scripts/entity_classifier_script_ACYC.py
python scripts/entity_classifier_script_COAD.py
python scripts/entity_classifier_script_ERMS.py  
python scripts/entity_classifier_script_SDCA.py
python scripts/entity_classifier_script_SDCA_MYEC.py  
python scripts/entity_classifier_script_PAAD.py
python scripts/entity_classifier_script_DSRCT.py

echo "All scripts finished at $(date)"