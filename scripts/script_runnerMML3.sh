#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_ACYC.py
python scripts/entity_classifier_script_COAD.py
python scripts/entity_classifier_script_DSRCT.py
python scripts/entity_classifier_script_ES.py
python scripts/entity_classifier_script_SYNS.py
python scripts/entity_classifier_script_SFT.py


echo "All scripts finished at $(date)"
