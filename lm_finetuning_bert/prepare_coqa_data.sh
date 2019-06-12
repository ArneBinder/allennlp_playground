#!/bin/sh

# Prepare CoQA data for language model train setting.

# do not forget to add/remove parameter "--do_lower_case" to python command, if necessary!
BERT_MODEL=bert-base-uncased

COQA_TRAIN_DATA=/home/abinder/datasets/CoQA/coqa-train-v1.0.json
TRAIN_PATH=training/coqa/$BERT_MODEL/


#cd allennlp_playground
# generate CoQA training data
mkdir -p $TRAIN_PATH/$BERT_MODEL
echo "prepare training data with: $COQA_TRAIN_DATA and write to: $TRAIN_PATH"
python lm_finetuning_bert/pregenerate_training_data.py \
--train_corpus $COQA_TRAIN_DATA \
--document_loader coqa_document_loader \
--bert_model $BERT_MODEL \
--do_lower_case \
--output_dir $TRAIN_PATH \
--epochs_to_generate 3 \
--max_seq_len 256