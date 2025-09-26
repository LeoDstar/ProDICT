#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_EHAE.py
python scripts/entity_classifier_script_EPIS.py

python scripts/entity_classifier_script_LGFMS.py
python scripts/entity_classifier_script_LUAD.py

python scripts/entity_classifier_script_MYEC.py
python scripts/entity_classifier_script_RMS.py
python scripts/entity_classifier_script_SCRMS.py



echo "All scripts finished at $(date)"
