#!/bin/sh

TRAIN_DIR=training/coqa/
MODEL_OUT_DIR=models/coqa_lm/

#cd allennlp_playground
python lm_finetuning_bert/finetune_on_pregenerated.py \
--pregenerated_data $TRAIN_DIR \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir $MODEL_OUT_DIR \
--epochs 3