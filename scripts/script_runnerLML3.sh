#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_BRCA.py
python scripts/entity_classifier_script_IHCH.py
python scripts/entity_classifier_script_ASPS.py
python scripts/entity_classifier_script_DIFG.py
python scripts/entity_classifier_script_PLEMESO_PEMESO.py
python scripts/entity_classifier_script_UM.py
python scripts/entity_classifier_script_MEL.py
python scripts/entity_classifier_script_MEL_UM.py
python scripts/entity_classifier_script_MPNST.py
python scripts/entity_classifier_script_COAD_READ.py


echo "All scripts finished at $(date)"
