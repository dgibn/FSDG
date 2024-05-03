import deeplake

train = deeplake.load("hub://activeloop/pacs-train")
val = deeplake.load("hub://activeloop/pacs-val")
test = deeplake.load("hub://activeloop/pacs-test")

print(train)

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root "/raid/biplab/divyam/Divyam/fsDG/pacs" \
--trainer DAELDG \
--source-domains photo sketch cartoon \
--target-domains art_painting \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/daeldg/pacs.yaml \
--output-dir output4/source_only_pacs 