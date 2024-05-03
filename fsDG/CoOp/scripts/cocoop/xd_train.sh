#!/bin/bash

cd ../..

# custom config
DATA=/raid/biplab/divyam/Divyam/fsDG/pacs
# TRAINER=CoCoOp
TRAINER=CoOp

DATASET=pacs
SEED=$1

# CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=8


DIR=/raid/biplab/divyam/Divyam/fsDG/CoOp/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python /raid/biplab/divyam/Divyam/fsDG/CoOp/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file /raid/biplab/divyam/Divyam/fsDG/CoOp/configs/datasets/${DATASET}.yaml \
    --config-file /raid/biplab/divyam/Divyam/fsDG/CoOp/configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi