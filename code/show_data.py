import pickle

def show_data(input_sentence_file, correct_sentence_file, input_pos_file, label_file, num_data):
    with open(input_sentence_file, 'rb') as file:
        sentences = pickle.load(file)
        file.close()
        print(sentences[:num_data])

    with open(correct_sentence_file, 'rb') as file:
        correction = pickle.load(file)
        file.close()
        print(correction[:num_data])

    with open(input_pos_file, 'rb') as file:
        input_pos = pickle.load(file)
        file.close()
        print(input_pos[:num_data])

    with open(label_file, 'rb') as file:
        errors = pickle.load(file)
        file.close()
        print(errors[:num_data])

    for word, pos, err_type in zip(sentences[:num_data], input_pos[:num_data], errors[:num_data]):
        print(f"{word}, {pos}, {err_type}")

    print(f"Input length: {len(sentences)}")
    print(f"POS length: {len(input_pos)}")
    print(f"Labels length: {len(errors)}")
    print(f"Correction length: {len(correction)}")

if __name__ == '__main__':
    # file location
    directory = '../processed_dataset/training/npltea16_HSK_TrainingSet/'
    input_sentence_file = directory + 'input_sentences'
    input_pos_file = directory + 'input_pos'
    correct_sentence_file = directory + 'correct_sentences'
    label_file = directory + 'errors'

    # start =
    # end =
    num_data = 100

    show_data(input_sentence_file, correct_sentence_file, input_pos_file, label_file, num_data)