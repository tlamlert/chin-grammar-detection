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
    :return: input_sentences,       a list of length <number of words in the input sentences>
            input_pos,              a list of length <number of words in the input sentences>
            correct_sentences,      a list of length <number of words in the correct sentences>
            errors                  a list of length <number of words in the input sentences>
    """
    # TODO: fix the function header
    # declare word parser and pos tagging models
    ws = WS("./data")           # word segmentation
    pos = POS("./data")         # pos tagging
    # ner = NER("./dataset")      # entity detection

    # TODO: Parse SGML into some variables
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
    input_pos = pos(raw_input_sentences)

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
    input_pos       = list(itertools.chain.from_iterable(input_pos))
    correct_sentences = list(itertools.chain.from_iterable(correct_sentences))
    corresponding_err = list(itertools.chain.from_iterable(corresponding_err))

    return input_sentences, input_pos, correct_sentences, corresponding_err

def save_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file,
             input_sentences, input_pos, correct_sentences, labels):
    '''
    save data to the given file locations
    '''
    with open(input_sentence_file, 'wb') as file:
        pickle.dump(input_sentences, file)
        file.close()

    with open(input_pos_file, 'wb') as file:
        pickle.dump(input_pos, file)
        file.close()

    with open(correct_sentence_file, 'wb') as file:
        pickle.dump(correct_sentences, file)
        file.close()

    with open(label_file, 'wb') as file:
        pickle.dump(labels, file)
        file.close()

def get_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file):
    '''
    retrieve and return data from the given file locations
    '''
    with open(input_sentence_file, 'rb') as file:
        sentences = pickle.load(file)
        file.close()

    with open(input_pos_file, 'wb') as file:
        input_pos = pickle.load(file)
        file.close()

    with open(correct_sentence_file, 'rb') as file:
        correction = pickle.load(file)
        file.close()

    with open(label_file, 'rb') as file:
        labels = pickle.load(file)
        file.close()

    return sentences, input_pos, correction, labels

if __name__ == '__main__':
    # file location
    to_be_processed_file = '../dataset/nlptea16cged_release1.0/Training/CGED16_HSK_TrainingSet.txt'
    # to_be_processed_file = 'test.txt'
    directory = '../processed_dataset/'
    input_sentence_file = directory + 'input_sentences'
    input_pos_file = directory + 'input_pos'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    print('processing...')
    start_time = time.time()
    # these return values are python lists
    input_sentences, input_pos, correct_sentences, labels = preprocess( to_be_processed_file )
    finish_time = time.time()

    print('saving data to files...')
    save_data(input_sentence_file, input_pos_file, correct_sentence_file, label_file,
              input_sentences, input_pos, correct_sentences, labels)

    assert(len(input_sentences) == len(labels))
    print(f"input length: {len(input_sentences)}")
    print(f"time to process: {finish_time - start_time}")
