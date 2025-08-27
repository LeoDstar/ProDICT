#!/bin/bash

echo "Starting scripts at $(date)"

python scripts/entity_classifier_script_OS.py
python scripts/entity_classifier_script_LMS.py
python scripts/entity_classifier_script_ULMS.py
python scripts/entity_classifier_script_LMS_ULMS.py
python scripts/entity_classifier_script_MRLS.py
python scripts/entity_classifier_script_DDLS.py
python scripts/entity_classifier_script_PANET.py


echo "All scripts finished at $(date)"
