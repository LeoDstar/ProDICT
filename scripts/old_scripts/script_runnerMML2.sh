#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_ARMS.py
python scripts/entity_classifier_script_ES.py
python scripts/entity_classifier_script_ACC.py
python scripts/entity_classifier_script_CHDM.py  
python scripts/entity_classifier_script_SYNS.py
python scripts/entity_classifier_script_SFT.py
python scripts/entity_classifier_script_IHCH.py
python scripts/entity_classifier_script_ASPS.py
python scripts/entity_classifier_script_DIFG.py

echo "All scripts finished at $(date)"