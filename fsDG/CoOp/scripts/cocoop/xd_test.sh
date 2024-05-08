#!/bin/bash

cd ../..

# custom config
DATA=/home/hassan/pacs_data
# TRAINER=CoCoOp
TRAINER=CoOp

DATASET=pacsnew
SEED=1

CFG=vit_b16_ep50_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=1
SOURCE=photo
TARGET=photo


DIR=/home/hassan/FSDG/fsDG/CoOp/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}_${SOURCE}_${TARGET}_shots_test/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --source-domains ${SOURCE} \
    --target-domains ${TARGET} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir /home/hassan/FSDG/fsDG/CoOp/output/pacsnew/${TRAINER}/${CFG}_${SHOTS}_${SOURCE}_${TARGET}_shots/seed${SEED} \
    --load-epoch 50 \
    --eval-only
fi