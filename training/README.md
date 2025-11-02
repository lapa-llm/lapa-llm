# Quickstart


```bash

pip install axolotl==0.12.1 torch=2.8.0

# Instruction run was trained on 3 nodes with 4 GPUs each
NUM_NODES=3
NUM_GPUS=4
torchrun \
--nnodes $NUM_NODES \
--nproc_per_node $NUM_GPUS \
-m axolotl.cli.train training/lapa-12b-instructions.yml ```