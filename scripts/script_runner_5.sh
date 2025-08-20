#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_BRCA.py
python scripts/entity_classifier_script_CHDM.py  
python scripts/entity_classifier_script_SYNS.py
python scripts/entity_classifier_script_LMS.py
python scripts/entity_classifier_script_ACYC.py
python scripts/entity_classifier_script_SFT.py


echo "All scripts finished at $(date)"