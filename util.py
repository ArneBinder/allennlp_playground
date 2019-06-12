import json
import logging
import re

import spacy
import plac

logger = logging.getLogger()
ch = logging.StreamHandler()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)  # This toggles all the logging in your app


def coqa_to_lm(file_name_in: ('input file name', 'option', 'i', str),
               file_name_out: ('output file name', 'option', 'o', str)):
    logger.info('load data from: %s' % file_name_in)
    data = json.load(open(file_name_in))

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #print(nlp.pipe_names)
    regex = re.compile(r"\n+")

    logger.info('write data to: %s...' % file_name_out)
    with open(file_name_out, 'w') as f:

        for story in nlp.pipe((record['story'] for record in data['data']), disable=['parser', 'tagger', 'ner'],
                              n_threads=4, batch_size=1000):
            for sent in story.sents:
                sent_text = sent.text
                sent_text = re.sub(regex, ' ', sent_text).strip()
                f.write(sent_text + '\n')
            f.write('\n')


if __name__ == '__main__':
    plac.call(coqa_to_lm)

              #(file_name_in='/mnt/DATA/ML/data/corpora/QA/CoQA/coqa-dev-v1.0.json',
              # file_name_out='/mnt/DATA/ML/data/corpora/QA/CoQA/stories_only/coqa-dev-v1.0.txt')
