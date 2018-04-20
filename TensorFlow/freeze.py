TF_TOOLS="/home/cucu/env/tensorflow/lib/python3.5/site-packages/tensorflow/python/tools/"

python3 $TF_TOOLS/freeze_graph.py \
  --input_graph=graph64.pb \
  --input_checkpoint=Model.ckpt.64 \
  --output_graph=frozen_graph.64.pb \
  --output_node_names=output/Softmax

python3 $TF_TOOLS/optimize_for_inference.py \
  --input frozen_graph.64.pb \
  --output opt_graph.64.pb \
  --frozen_graph True \
  --input_names input/Identity \
  --output_names output/Softmax

python3 $TF_TOOLS/freeze_graph.py \
  --input_graph=graph1.pb \
  --input_checkpoint=Model.ckpt.1 \
  --output_graph=frozen_graph.1.pb \
  --output_node_names=output/Softmax

python3 $TF_TOOLS/optimize_for_inference.py \
  --input frozen_graph.1.pb \
  --output opt_graph.1.pb \
  --frozen_graph True \
  --input_names input/Identity \
  --output_names output/Softmax


#~/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
#  --in_graph=opt_graph.pb \
#  --out_graph=fused_graph.pb \
#  --inputs=input \
#  --outputs=regression_output/BiasAdd \
#  --transforms="fold_constants sort_by_execution_order"
