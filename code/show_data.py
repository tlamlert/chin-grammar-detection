import pickle

def get_data(input_sentence_file, correct_sentence_file, label_file):
    with open(input_sentence_file, 'rb') as file:
        sentences = pickle.load(file)
        file.close()

    with open(correct_sentence_file, 'rb') as file:
        correction = pickle.load(file)
        file.close()

    with open(label_file, 'rb') as file:
        errors = pickle.load(file)
        file.close()

    return sentences, errors, correction

if __name__ == '__main__':
    # file location
    directory = '../processed_dataset/'
    input_sentence_file = directory + 'input_sentences'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    # get data
    sentences, errors, correction = get_data(input_sentence_file, correct_sentence_file, label_file)

    # print data
    num_data = 100
    print(sentences[:num_data])
    print(errors[:num_data])
    print(correction[:num_data])
    for word, err_type in zip(sentences[:num_data], errors[:num_data]):
        print(f"{word}, {err_type}")