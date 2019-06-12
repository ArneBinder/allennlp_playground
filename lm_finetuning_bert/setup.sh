#!/bin/sh

COQA_TRAINDATA_PATH=/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json

# create conda environment with required packages
conda create -n allennlp_playground -c allennlp python=3 pytorch-pretrained-bert spacy
# activate conda environment
source activate allennlp_playground
# download spacy model
python -m spacy download en_core_web_sm

# move into directory
cd allennlp_playground
