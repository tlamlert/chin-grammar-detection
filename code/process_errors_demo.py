def main():
    input_sentences = [['别', '只', '能', '想', '自己', '，', '想', '你', '周围', '的', '人', '。', '还', '有', '你', '，', '如果', '你', '是', '一', '个', '家庭', '的', '爸爸', '，', '你', '多', '想', '自己', '的', '孩子', '；', '如果', '你', '是', '青少年', '你', '多', '想', '自己', '的', '未', '来', '；', '那', '你', '可以', '禁烟', '了', '。'],
                       ['若', '父母', '主动', '和', '孩子', '平起平坐', '，', '倾心交谈', '，', '互相', '充分', '理解', '对方', '，', '会', '能够', '大幅', '减', '少', '互相', '抱怨', '的', '情况', '。'],
                       ['到底', '是', '健康', '重要', '，', '还', '是', '粮食产生量', '重要', '呢', '？', '对这个问题', '的', '我', '的', '意见', '是', '产生量', '更', '重要', '。']]
    error_list = [[('3', '3', 'R'), ('8', '8', 'M'), ('9', '9', 'M'), ('17', '17', 'R'), ('32', '32', 'M'), ('34', '34', 'M'), ('48', '48', 'M'), ('50', '50', 'M'), ('58', '58', 'M'), ('60', '60', 'S')],
                  [('29', '30', 'R')],
                  [('13', '14', 'S'), ('25', '25', 'R')]]

    list_of_errors = []
    for sentence, errors in zip(input_sentences, error_list):
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
                # print(f"{start}, {end}, {st_err}, {ed_err}, {word}")
                if ((start <= ed_err) & (end >= st_err)):
                    errs.append(err_type)
                    if end + 1 >= ed_err:
                        num_err += 1
                else:
                    errs.append("C")
            else:
                errs.append("C")
        print(errs)
        # print(f"{len(sentence)}, {len(errs)}")
        list_of_errors.append(errs)

if __name__ == '__main__':
    main()