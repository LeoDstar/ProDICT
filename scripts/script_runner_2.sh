#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_SDCA.py
python scripts/entity_classifier_script_SDCA_MYEC.py  
python scripts/entity_classifier_script_MEL.py
python scripts/entity_classifier_script_THYC.py
python scripts/entity_classifier_script_GIST.py
python scripts/entity_classifier_script_DIFG.py
python scripts/entity_classifier_script_COAD.py
python scripts/entity_classifier_script_MPNST.py

echo "All scripts finished at $(date)"