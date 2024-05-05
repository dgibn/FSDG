#!/bin/bash

cd ../..

# custom config
DATA=/home/hassan/pacs_data
# TRAINER=CoCoOp
TRAINER=CoOp

DATASET=pacsnew
SEED=1

# CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=1

rm -rf /home/hassan/FSDG/fsDG/CoOp/output/pacsnew
rm -rf /home/hassan/pacs_data/split_fewshot_art_painting
rm -rf /home/hassan/pacs_data/split_fewshot_sketch
rm /home/hassan/pacs_data/split_pacs_art_painting.json
rm /home/hassan/pacs_data/split_pacs_sketch.json
DIR=/home/hassan/FSDG/fsDG/CoOp/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python /home/hassan/FSDG/fsDG/CoOp/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file /home/hassan/FSDG/fsDG/CoOp/configs/datasets/${DATASET}.yaml \
    --config-file /home/hassan/FSDG/fsDG/CoOp/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi