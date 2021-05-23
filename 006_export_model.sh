#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Did you update the 'tensorflow_models' path in this script?"
read -rsp $'Press any key to continue...\n' -n 1 key
echo ">>>>>"

python 006_export_model.py --input_type image_tensor --pipeline_config_path ./models/mono_ssd_mobilenet_v2_fplite/pipeline.config --trained_checkpoint_dir ./models/mono_ssd_mobilenet_v2_fplite/ --output_directory ./exported-models/prueba

# Output:
# Converted 199 variables to const ops.
python 006_export_model.py --input_type image_tensor --pipeline_config_path ./models/mono_ssd_mobilenet_v2_fplite/pipeline.config --trained_checkpoint_dir ./models/mono_ssd_mobilenet_v2_fplite/ --output_directory ./exported-models/Sheet_TL_Detection
