import numpy as np
from show_data import get_data

def window_data(inputs, window_sz):
    '''
    'window a 1-dimensional list into a 2-dimensional numpy array of shape [window_num, window_sz]
    :param inputs:      a list of elements
    :param window_sz:   the size of windows
    :return: windowed_inputs:       a 2-d numpy array of shape [window_num, window_sz]
    '''
    num_inputs = len(inputs)
    # ignore what doesn't fit into the window size
    inputs = inputs[:num_inputs - num_inputs % window_sz]

    # reshape into np of shape [window_num, window_sz]
    windowed_inputs = np.reshape(inputs, (-1, window_sz))

    return windowed_inputs

if __name__ == '__main__':
    # file location
    directory = '../processed_dataset/'
    input_sentence_file = directory + 'input_sentences'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    inputs, labels, _ = get_data(input_sentence_file, correct_sentence_file, label_file)
    window_size = 20
    windowed_inputs = window_data(inputs, window_size)
    windowed_labels = window_data(labels, window_size)

    print(windowed_inputs.shape)
    print(windowed_labels.shape)