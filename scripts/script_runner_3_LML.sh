#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_DSRCT.py
python scripts/entity_classifier_script_PANET.py  
python scripts/entity_classifier_script_MRLS.py
python scripts/entity_classifier_script_DDLS.py
python scripts/entity_classifier_script_PAAD.py
python scripts/entity_classifier_script_ASPS.py


echo "All scripts finished at $(date)"