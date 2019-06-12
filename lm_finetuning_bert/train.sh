#!/bin/sh

# Train (fine-tune) BERT on GPUs (e.g. with prepared CoQA data).

TRAIN_DIR=training/coqa/bert-base-uncased/
MODEL_OUT_DIR=models/coqa/bert-base-uncased/

# use first argument as GPU devices, if available
if [ -n "$1" ]; then
    PY=CUDA_VISIBLE_DEVICES=$1 python
    echo "use CUDA_VISIBLE_DEVICES=$1"
else
    PY=python
    echo "WARNING: CUDA_VISIBLE_DEVICES is not set!"
fi

#cd allennlp_playground
mkdir -p $MODEL_OUT_DIR
$PY lm_finetuning_bert/finetune_on_pregenerated.py \
--pregenerated_data $TRAIN_DIR \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir $MODEL_OUT_DIR \
--epochs 3