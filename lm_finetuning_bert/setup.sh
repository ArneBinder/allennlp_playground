#!/bin/sh

COQA_TRAINDATA=coqa-train-v1.0.json
COQA_PATH=/home/abinder/datasets/CoQA/
CONDA_ENV=allennlp_playground

echo "download CoQA train data: https://nlp.stanford.edu/data/coqa/$COQA_TRAINDATA to $COQA_PATH ..."
cd $COQA_PATH
wget https://nlp.stanford.edu/data/coqa/$COQA_TRAINDATA

echo "create conda environment with required packages (pytorch-pretrained-bert, pytorch and spacy): $CONDA_ENV ..."
conda create -n $CONDA_ENV -c conda-forge -c pytorch python=3 "blas=*=mkl" pytorch-pretrained-bert pytorch spacy
echo "activate conda environment: $CONDA_ENV"
conda activate $CONDA_ENV
echo "download spacy model ..."
python -m spacy download en_core_web_sm

# move into directory
cd allennlp_playground
