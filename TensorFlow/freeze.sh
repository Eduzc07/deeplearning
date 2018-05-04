#!/bin/bash

TF_TOOLS="/home/cucu/env/tensorflow/lib/python3.5/site-packages/tensorflow/python/tools/"

python3 $TF_TOOLS/freeze_graph.py \
  --input_graph=$1 \
  --input_checkpoint=$2 \
  --output_graph=frozen_$1.pb \
  --output_node_names=output/Softmax

python3 $TF_TOOLS/optimize_for_inference.py \
  --input frozen_$1.pb \
  --output infer_$1.pb \
  --frozen_graph True \
  --input_names input/Identity \
  --output_names output/Softmax

