#!/bin/sh

# Train (fine-tune) BERT on GPUs (e.g. with prepared CoQA data).

TRAIN_DIR=training/coqa/bert-base-uncased/
MODEL_OUT_DIR=models/coqa/bert-base-uncased/

GPUS=0,1

#cd allennlp_playground
mkdir -p $MODEL_OUT_DIR
CUDA_VISIBLE_DEVICES=$GPUS python lm_finetuning_bert/finetune_on_pregenerated.py \
--pregenerated_data $TRAIN_DIR \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir $MODEL_OUT_DIR \
--epochs 3