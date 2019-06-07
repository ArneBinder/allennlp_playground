import os
from typing import Iterator, List, Dict, Callable, Iterable
import json

import torch
import torch.optim as optim
import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.data import Instance
from allennlp.data.dataset_readers import LanguageModelingReader, SimpleLanguageModelingDatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.models import BidirectionalLanguageModel

import logging
#logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

DEBUG = True


# see https://github.com/allenai/allennlp/blob/e6ad6e9a90b55f76dc921b35ec28578d82af8bbd/allennlp/tests/fixtures/language_model/experiment_unsampled.jsonnet
def create_model(vocab):
    # prepare model
    EMBEDDING_DIM = 100
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    HIDDEN_DIM = 100
    bilstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    model = BidirectionalLanguageModel(vocab=vocab, text_field_embedder=word_embeddings, contextualizer=bilstm)
    return model


def train(model_dir):

    # prepare data
    #reader = CoqaDatasetReader()
    #reader = CoqaDatasetReader(tokenizer=lambda x: WordTokenizer().tokenize(text=x))
    #reader = LanguageModelingReader(tokenizer=WordTokenizer(word_splitter=SpacyWordSplitter(language='en_core_web_sm')))
    reader = SimpleLanguageModelingDatasetReader(tokenizer=WordTokenizer(word_splitter=SpacyWordSplitter(language='en_core_web_sm')))
    train_dataset = reader.read(cached_path(
        '/mnt/DATA/ML/data/corpora/QA/CoQA/stories_only/coqa-train-v1.0_extract100.json'))
    validation_dataset = reader.read(cached_path(
        '/mnt/DATA/ML/data/corpora/QA/CoQA/stories_only/coqa-dev-v1.0.json'))

    vocab = None
    model_fn = os.path.join(model_dir, 'model.th')
    vocab_fn = os.path.join(model_dir, 'vocab')
    if os.path.exists(model_dir):
        if os.path.exists(vocab_fn):
            logging.info('load vocab from: %s...' % vocab_fn)
            vocab = Vocabulary.from_files(vocab_fn)
    else:
        os.makedirs(model_dir)
    if vocab is None:
        #vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
        vocab = Vocabulary.from_instances(train_dataset)
        #TODO: re-add!
        #vocab.extend_from_instances(validation_dataset)
        logging.info('save vocab to: %s...' % vocab_fn)
        vocab.save_to_files(vocab_fn)
    logging.info('data prepared')

    model = create_model(vocab)

    if os.path.exists(model_fn):
        logging.info('load model wheights from: %s...' % model_fn)
        with open(model_fn, 'rb') as f:
            model.load_state_dict(torch.load(f))
    logging.info('model prepared')

    # prepare training
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=10)
    logging.info('training prepared')

    trainer.train()

    logging.info('save model to: %s...' % model_fn)
    with open(model_fn, 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    train('models/lm')
    #generate('models/lm')