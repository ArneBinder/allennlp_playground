import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def get_hidden(tokens_tensor):
    # Load pre-trained model (weights)
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    model.eval()

    # If you have a GPU, put everything on cuda
    if torch.cuda.is_available():
        tokens_tensor = tokens_tensor.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # Predict hidden states features for each layer
        hidden_states, mems = model(tokens_tensor)
        # We can re-use the memory cells in a subsequent call to attend a longer context
        #hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)
    return hidden_states


def predict_token(tokens_tensors):
    # Load pre-trained model (weights)
    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    model.eval()

    # If you have a GPU, put everything on cuda
    if torch.cuda.is_available():
        for i, tt in enumerate(tokens_tensors):
            tokens_tensors[i] = tokens_tensors[i].to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # Predict all tokens
        mems = None
        all_predictions = []
        for i, tt in enumerate(tokens_tensors):
            #predictions, mems = model(tt)
            # We can re-use the memory cells in a subsequent call to attend a longer context
            #predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)
            predictions, mems = model(tt, mems=mems)
            all_predictions.append(predictions)


    # get the predicted last token
    predicted_index = torch.argmax(all_predictions[-1][0, -1, :]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #assert predicted_token == 'who'
    return predicted_token


if __name__ == '__main__':

    # Load pre-trained model tokenizer (vocabulary from wikitext 103)
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

    tokenizer.add_symbol('@@@BACKGROUND@@@')
    tokenizer.add_symbol('@@@QUESTION@@@')
    tokenizer.add_symbol('@@@ANSWER@@@')
    # debug save modified vocab
    open('models/transfxl.vocab', 'w').writelines((t + '\n' for t in tokenizer.idx2sym))

    # Tokenized input
    #text_1 = "Who was Jim Henson ?"
    text_1 = "@@@QUESTION@@@ Who was Jim Henson ? Jim Henson was a puppeteer ."
    text_2 = "Jim Henson was a puppeteer"
    tokenized_text_1 = tokenizer.tokenize(text_1)
    tokenized_text_2 = tokenizer.tokenize(text_2)

    # Convert token to vocabulary indices
    indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
    indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

    tokenized_text_1_dummy = tokenizer.convert_ids_to_tokens(indexed_tokens_1)[0]

    # Convert inputs to PyTorch tensors
    tokens_tensor_1 = torch.tensor([indexed_tokens_1])
    tokens_tensor_2 = torch.tensor([indexed_tokens_2])

    #print(predict_token([tokens_tensor_1, tokens_tensor_2]))
