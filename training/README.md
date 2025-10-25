# Quickstart


`pip install axolotl==0.12.1

torchrun \
--nnodes $NUM_NODES \
--nproc_per_node $NUM_GPUS
-m axolotl.cli.train training/lapa-12b-instructions.yml `