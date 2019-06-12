import spacy

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path.split('/')[-1])
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, context, questions, answers):

    # Load pre-trained model tokenizer (vocabulary)
    #tokenizer = BertTokenizer.from_pretrained(model_path.split('/')[-1])
    #tokenizer = BertTokenizer.from_pretrained(model_path)

    # Tokenized input
    #text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    #tokenized_text = tokenizer.tokenize(context)
    tokenized_text = ['[CLS]', '[unused0]']
    for sent in context:
        tokenized_text.extend(tokenizer.tokenize(sent))

    assert len(questions) == len(answers), 'length of questions / answer mismatch'

    for i in range(len(questions)):
        tokenized_text.extend(['[unused1]'] + tokenizer.tokenize(questions[i]))
        tokenized_text.extend(['[unused2]'] + tokenizer.tokenize(answers[i]))

    print(tokenized_text)
    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 18
    tokenized_text[masked_index] = '[MASK]'
    #assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
    #debug
    print(tokenized_text)

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

    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #assert predicted_token == 'henson'
    return predicted_token


if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer(model_path='/mnt/DATA/DEVELOPING/PycharmProjects/allennlp_playground/models/coqa/bert-base-uncased')

    context_str = "Jim Henson was a puppeteer."
    questions = ['Was Jim Henson a puppeteer?']
    answers = ['yes']
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    context_doc = nlp(context_str.strip(), disable=['parser', 'tagger', 'ner'])
    context = [sent.text for sent in context_doc.sents]

    print(predict(model=model, tokenizer=tokenizer, context=context, questions=questions, answers=answers))
