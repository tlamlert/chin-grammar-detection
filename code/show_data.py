import pickle

def show_data(input_sentence_file, correct_sentence_file, label_file, num_data):
    with open(input_sentence_file, 'rb') as file:
        sentences = pickle.load(file)
        file.close()
    for sentence in sentences[:num_data]:
        print(sentence)

    with open(correct_sentence_file, 'rb') as file:
        correction = pickle.load(file)
        file.close()
    for sentence in correction[:num_data]:
        print(sentence)

    with open(label_file, 'rb') as file:
        errors = pickle.load(file)
        file.close()
    print(errors[:num_data])

    # for sentence, errs in zip(sentences, errors):
    #     for word, err_type in zip(sentence, errs):
    #         print(f"{word}, {err_type}")
    #     print()

if __name__ == '__main__':
    # file location
    directory = '../processed_dataset/'
    input_sentence_file = directory + 'input_sentences'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    num_data = 5

    show_data(input_sentence_file, correct_sentence_file, label_file, num_data)