#!/bin/sh

# Train (fine-tune) BERT on GPUs (e.g. with prepared CoQA data).

TRAIN_DIR=training/coqa/bert-base-uncased/
MODEL_OUT_DIR=models/coqa/bert-base-uncased/

# use first argument as GPU devices, if available
if [ -n "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "use CUDA_VISIBLE_DEVICES=$1"
else
    echo "WARNING: CUDA_VISIBLE_DEVICES is not set!"
fi

CMD="python lm_finetuning_bert/finetune_on_pregenerated.py"
echo "execute: $CMD"

#cd allennlp_playground
mkdir -p $MODEL_OUT_DIR
$CMD \
--pregenerated_data $TRAIN_DIR \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir $MODEL_OUT_DIR \
--epochs 3