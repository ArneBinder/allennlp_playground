import os
from typing import Iterator, List, Dict, Callable, Iterable
import json

import torch
import torch.optim as optim
import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.data import Instance
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

import logging
#logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

DEBUG = True


def tonp(tsr): return tsr.detach().cpu().numpy()


@DatasetReader.register('coqa')
class CoqaDatasetReader(DatasetReader):
    """
    DatasetReader for CoQA data, see https://stanfordnlp.github.io/coqa/

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Callable[[str], List[Token]]=lambda x: [Token(t) for t in x.split()]) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], labels: List[Token] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if labels:
            label_field = TextField(tokens, self.token_indexers)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        #with open(file_path) as f:
        #    for line in f:
        #        pairs = line.strip().split()
        #        sentence, tags = zip(*(pair.split("###") for pair in pairs))
        #        yield self.text_to_instance([Token(word) for word in sentence], tags)

        with open(file_path) as f:
            data = json.load(f)
            n = 0
            for record in data['data']:
                context = record['story']
                sentence = self.tokenizer(context)
                yield self.text_to_instance(tokens=[Token('@@SOS@@')] + sentence, labels=sentence + [Token('@@EOS@@')])
                n += 1
                if DEBUG and n==10:
                    break


@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None
                ) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels['tokens'], mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels['tokens'], mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# not used
class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        #self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        #values, indices = out_dict["tag_logits"].max(axis=-1)
        token_ids = np.argmax(tonp(out_dict["tag_logits"]), axis=-1)
        #return expit(tonp(out_dict["tag_logits"]))
        return token_ids

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                #batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


def create_model(vocab):
    # prepare model
    EMBEDDING_DIM = 100
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    HIDDEN_DIM = 100
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)
    return model


def generate(model_dir, context='@@SOS@@ The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. \n\nThe Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. \n\nIn March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. \n\nThe Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. \n\nScholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. \n\nThe Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.'):
    assert os.path.exists(model_dir), '%s not found' % model_dir
    model_fn = os.path.join(model_dir, 'model.th')
    vocab_fn = os.path.join(model_dir, 'vocab')
    logging.info('load vocab from: %s...' % vocab_fn)
    vocab = Vocabulary.from_files(vocab_fn)
    model = create_model(vocab)
    with open(model_fn, 'rb') as f:
        model.load_state_dict(torch.load(f))

    ## iterate over the dataset without changing its order
    #seq_iterator = BasicIterator(batch_size=64)
    #seq_iterator.index_with(vocab)
    #predictor = Predictor(model, seq_iterator)

    reader = CoqaDatasetReader(tokenizer=lambda sent: SpacyWordSplitter(language='en_core_web_sm').split_words(sent))
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict(context)['tag_logits']
    token_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'tokens') for i in token_ids])


def train(model_dir):

    # prepare data
    #reader = CoqaDatasetReader()
    #reader = CoqaDatasetReader(tokenizer=lambda x: WordTokenizer().tokenize(text=x))
    reader = CoqaDatasetReader(tokenizer=lambda sent: SpacyWordSplitter(language='en_core_web_sm').split_words(sent))
    train_dataset = reader.read(cached_path(
        '/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json'))
    validation_dataset = reader.read(cached_path(
        '/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-dev-v1.0.json'))

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
        vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
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
    iterator = BasicIterator(batch_size=2)
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=100)
    logging.info('training prepared')

    trainer.train()

    logging.info('save model to: %s...' % model_fn)
    with open(model_fn, 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    #train('models/lmqa_coqa')
    generate('models/lmqa_coqa')
