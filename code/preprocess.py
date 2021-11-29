import tensorflow as tf
import numpy as np
import pickle
from bs4 import BeautifulSoup
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def preprocess(file_name):
    """
    Preprocess processes the input dataset which comes in SGML format
    :param file_name: location of the file to be preprocessed
    :return: input_sentences,       a list of tensors of shape [sentence_size, 2]
            correct_sentences,      a list of tensors of shape [sentence_size, 2]
            errors                  a list of tensors of shape [sentence_size, error_class]
    """
    # TODO: fix the function header
    # declare word parser and pos tagging models
    ws = WS("./data")           # word segmentation
    pos = POS("./data")         # pos tagging
    # ner = NER("./dataset")      # entity detection

    # TODO: Parse SGML into something
    raw_input_sentences = []     # list of sentences
    raw_correct_sentences = []   # list of sentences
    list_of_errors = []
    with open(file_name, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')

    docs = soup.find_all('doc')  # find all object with tag 'doc'
    for doc in docs:
        text = doc.find('text').string.strip()
        correction = doc.find('correction').string.strip()
        raw_errors = doc.find_all('error')
        errors = []
        for err in raw_errors:
            tuple = (err['end_off'], err['start_off'], err['type'])
            errors.append(tuple)

        raw_input_sentences.append(text)
        raw_correct_sentences.append(correction)
        list_of_errors.append(errors)

    # TODO: Apply ckiptagger to parse the raw input sentences
    input_sentences = ws(raw_input_sentences)
    input_pos = pos(raw_input_sentences)
    input_sentences = np.append(input_sentences, input_pos, axis=0)

    correct_sentences = ws(raw_correct_sentences)
    correct_pos = pos(raw_correct_sentences)
    correct_sentences = np.append(correct_sentences, correct_pos, axis=0)

    return input_sentences, correct_sentences, list_of_errors

def main():
    input_sentences, correct_sentences, list_of_errors = preprocess('test.txt')

    with open('input_sentences.txt', 'wb') as file:
        pickle.dump(input_sentences, file)
        file.close()

    with open('correct_sentences.txt', 'wb') as file:
        pickle.dump(correct_sentences, file)
        file.close()

    with open('errors.txt', 'wb') as file:
        pickle.dump(list_of_errors, file)
        file.close()

if __name__ == '__main__':
    main()