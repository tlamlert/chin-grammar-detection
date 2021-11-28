import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def preprocess(raw_data):
    """
    Preprocess processes the input dataset which comes in SGML format
    :param raw_data: raw dataset in SGML format
    :return: input sentences,       a list of tensors of shape [sentence_size, 2]
            corrected sentences,    a list of tensors of shape [sentence_size, 2]
            errors                  a list of tensors of shape [sentence_size, error_class]
    sentence_size represents the number of words in a sentence
    """
    # declare word parser and pos tagging models
    ws = WS("./data")           # word segmentation
    pos = POS("./data")         # pos tagging
    # ner = NER("./dataset")      # entity detection

    # TODO: Parse SGML into something
    raw_input_sentences = []     # list of sentences
    raw_correct_sentences = []   # list of sentences

    # TODO: Apply ckiptagger to parse the raw input sentences
    input_sentences = ws(raw_input_sentences)
    input_pos = pos(raw_input_sentences)

    correct_sentences = ws(raw_correct_sentences)
    correct_pos = pos(raw_correct_sentences)

    errors

    return input_sentences, correct_sentences, errors