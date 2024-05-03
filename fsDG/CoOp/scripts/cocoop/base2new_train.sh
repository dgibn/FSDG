#!/bin/bash

cd ../..

# custom config
DATA=/raid/biplab/divyam/Divyam/fsDG/pacs
# TRAINER=CoCoOp
TRAINER=CoOp

# CFG=vit_b16_c4_ep10_batch1_ctxv1
# # CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# # CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
# SHOTS=16


DIR=/raid/biplab/divyam/Divyam/fsDG/CoOp/output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=1


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
    --shots ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi