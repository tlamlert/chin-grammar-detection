from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def main():
    ws = WS("./data")           # word segmentation
    pos = POS("./data")         # pos tagging

    # example sentences
    raw_sentences = [
        '别只能想自己，想你周围的人。还有你，如果你是一个家庭的爸爸，你多想自己的孩子；如果你是青少年你多想自己的未来；那你可以禁烟了。',
        '别只想自己，要想想你周围的人。还有，如果你是一个家庭的爸爸，你要多想想自己的孩子；如果你是青少年你要多想想自己的未来；那你就可以戒烟了'
    ]

    word_sentences = ws(raw_sentences)
    pos_sentences = pos(raw_sentences)

    # print results
    for sentence, words, pos in zip(raw_sentences, word_sentences, pos_sentences):
        print(f"{sentence}")
        for word, pos in zip(words, pos):
            print(f"{word}({pos}) ", end=' ')
        print()

if __name__ == '__main__':
    main()