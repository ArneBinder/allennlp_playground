import json

import spacy
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path.split('/')[-2])
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, context, questions, answers, sentencizer, masked_indices=None):

    # Load pre-trained model tokenizer (vocabulary)
    #tokenizer = BertTokenizer.from_pretrained(model_path.split('/')[-1])
    #tokenizer = BertTokenizer.from_pretrained(model_path)

    # Tokenized input
    #text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    #tokenized_text = tokenizer.tokenize(context)
    tokenized_text = ['[CLS]', '[unused0]']
    for sent in sentencizer(context):
        tokenized_text.extend(tokenizer.tokenize(sent))

    assert len(questions) == len(answers), 'length of questions / answer mismatch'

    for i in range(len(questions)):
        tokenized_text.extend(['[unused1]'] + tokenizer.tokenize(questions[i]))
        if type(answers[i]) == list:
            tokenized_text.extend(['[unused2]'] + answers[i])
        else:
            tokenized_text.extend(['[unused2]'] + tokenizer.tokenize(answers[i]))

    print('tokenized_text:\n%s' % tokenized_text)
    tokenized_text = np.array(tokenized_text)
    # Mask a token that we will try to predict back with `BertForMaskedLM`
    #masked_index = 18
    if masked_indices is not None:
        masked_indices = np.array(masked_indices, dtype=int)
        print('mask tokens:\n%s' % tokenized_text[masked_indices])
        tokenized_text[masked_indices] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    #segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    #model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    #model = BertForMaskedLM.from_pretrained(model_path)
    #model.eval()

    # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #segments_tensors = segments_tensors.to('cuda')
    #model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    # select all masked tokens
    masked_indices = np.nonzero(tokenized_text == '[MASK]')[0]

    # confirm we were able to predict 'henson'
    masked_predictions = predictions[0, masked_indices]
    predicted_indices = torch.argmax(masked_predictions, dim=-1).detach().cpu().numpy()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)
    #assert predicted_token == 'henson'
    return predicted_tokens


def coqa_iterator(fn):
    data = json.load(open(fn))['data']

    for record in data:
        context = record['story']
        questions = [x['input_text'] for x in record['questions']]
        answers = [x['input_text'] for x in record['answers']]
        yield context, questions, answers



if __name__ == '__main__':

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    sentencizer = lambda s: [sent.text for sent in nlp(s.strip(), disable=['parser', 'tagger', 'ner']).sents]
    model, tokenizer = load_model_and_tokenizer(model_path='/mnt/DATA/DEVELOPING/PycharmProjects/allennlp_playground/models/coqa/bert-base-uncased/epochs_20')

    #context = "The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. \n\nThe Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. \n\nIn March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. \n\nThe Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. \n\nScholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. \n\nThe Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant."
    #questions = ['When was the Vat formally opened?']

    # if any answer is already a list, it won't be tokenized
    #answers = [['[MASK]'] * 4]
    # this requires to hand to predict e.g. masked_indices=[-3, -2, -1]
    #answers = ['in 1475']

    for i, (context, questions, answers) in enumerate(coqa_iterator('/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-train-v1.0.json')):
        # taking all causes exception (?)
        questions = questions[:7]
        answers = answers[:7]
        for j in range(len(questions)):
            print('question:\n%s' % questions[-1])
            correct_answer = answers.pop()
            print('correct answer:\n%s' % correct_answer)
            answers.append(['[MASK]'] * len(correct_answer.split()))
            predicted_tokens = predict(model=model, tokenizer=tokenizer, context=context, questions=questions, answers=answers,
                                       sentencizer=sentencizer
                                       #masked_indices=[-3, -2, -1]
                                       )
            print('predicted tokens:\n%s' % predicted_tokens)
            answers.pop()
            questions.pop()

        break
