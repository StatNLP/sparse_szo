#!/bin/bash

# usage:
# $ bash ./run.sh 123 cifar10 fc3


seed=$1
data=$2
model=$3


for prune_or_freeze in prune freeze
do
    for masking_strategy in random heldout L1
    do
        python ./szo/run.py \
		    --seed=$seed \
		    --opt=agarwal \
		    --model=$model \
		    --data=$data \
		    --reward=nce \
		    --prune_or_freeze=$prune_or_freeze \
		    --masking_strategy=$masking_strategy \
		    --init=last \
		    --pr=0.2 \
		    --num_epochs=5 \
	      --num_rounds=20 \
        --num_samples=10 \
		    --batch_size=64 \
		    --eval_interval=10000 \
		    --eval_batch_size=10000 \
		    --lr=0.02 \
		    --var=1.0 \
		    --mu=0.05 \
        --beta=0.0 \
        --max_grad_norm=0.0 \
		    --device=gpu
    done
done