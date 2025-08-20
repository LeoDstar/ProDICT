#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_SDCA_MYEC.py
python scripts/entity_classifier_script_BA_ANGS.py  
python scripts/entity_classifier_script_IHCH.py
python scripts/entity_classifier_script_MEL.py
python scripts/entity_classifier_script_PAAD.py
python scripts/entity_classifier_script_PANET.py
python scripts/entity_classifier_script_GINET.py
python scripts/entity_classifier_script_MFH.py


echo "All scripts finished at $(date)"