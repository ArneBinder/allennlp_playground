#!/bin/sh

COQA_TRAIN_DATA=/home/abinder/datasets/CoQA/coqa-train-v1.0.json
TRAIN_PATH=training/coqa/

#cd allennlp_playground
# generate CoQA training data
mkdir -p $TRAIN_PATH
echo "prepare training data with: $COQA_TRAIN_DATA and write to: $TRAIN_PATH"
python lm_finetuning_bert/pregenerate_training_data.py --train_corpus $COQA_TRAIN_DATA --document_loader coqa_document_loader --bert_model bert-base-uncased --do_lower_case --output_dir $TRAIN_PATH --epochs_to_generate 3 --max_seq_len 256