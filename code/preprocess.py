# import tensorflow as tf
import numpy as np
import pickle
import itertools
import time
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
    # pos = POS("./data")         # pos tagging
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
            tuple = (err['start_off'], err['end_off'], err['type'])
            errors.append(tuple)

        raw_input_sentences.append(text)
        raw_correct_sentences.append(correction)
        list_of_errors.append(errors)

    # TODO: Apply ckiptagger to parse the raw input sentences
    input_sentences = ws(raw_input_sentences)
    # input_pos = pos(raw_input_sentences)
    # input_sentences = np.append(input_sentences, input_pos, axis=0)

    correct_sentences = ws(raw_correct_sentences)

    # TODO: parse and pad the list of errors
    corresponding_err = []
    for sentence, errors in zip(input_sentences, list_of_errors):
        start, end = 0, 0
        num_err = 0
        errs = []
        for word in sentence:
            start = end + 1
            end = end + len(word)
            if (num_err < len(errors)):
                st_err = int(errors[num_err][0])
                ed_err = int(errors[num_err][1])
                err_type = errors[num_err][2]
                if ((start <= ed_err) & (end >= st_err)):
                    errs.append(err_type)
                    if end + 1 >= ed_err:
                        num_err += 1
                else:
                    errs.append("C")
            else:
                errs.append("C")
        corresponding_err.append(errs)

    # TODO: concatenate return values into one single list
    input_sentences = list(itertools.chain.from_iterable(input_sentences))
    correct_sentences = list(itertools.chain.from_iterable(correct_sentences))
    corresponding_err = list(itertools.chain.from_iterable(corresponding_err))

    # print(input_sentences)
    # print(correct_sentences)
    # print(corresponding_err)

    return input_sentences, correct_sentences, corresponding_err

def main():
    # file location
    to_be_processed_file = '../dataset/nlptea16cged_release1.0/Training/CGED16_HSK_TrainingSet.txt'
    # to_be_processed_file = 'test.txt'

    # file location
    directory = '../processed_dataset/'
    input_sentence_file = directory + 'input_sentences'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    print('processing...')
    start_time = time.time()
    input_sentences, correct_sentences, corresponding_err = preprocess( to_be_processed_file )
    finish_time = time.time()

    print('saving data to files...')
    with open(input_sentence_file, 'wb') as file:
        pickle.dump(input_sentences, file)
        file.close()

    with open(correct_sentence_file, 'wb') as file:
        pickle.dump(correct_sentences, file)
        file.close()

    with open(label_file, 'wb') as file:
        pickle.dump(corresponding_err, file)
        file.close()

    assert(len(input_sentences) == len(corresponding_err))
    print(f"input length: {len(input_sentences)}")
    print(f"time to process: {finish_time - start_time}")

if __name__ == '__main__':
    main()
