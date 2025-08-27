#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_ANGS.py
python scripts/entity_classifier_script_THYM.py
python scripts/entity_classifier_script_THYC.py
python scripts/entity_classifier_script_CHS.py
python scripts/entity_classifier_script_DFSP.py
python scripts/entity_classifier_script_GIST.py
python scripts/entity_classifier_script_CCS.py


echo "All scripts finished at $(date)"