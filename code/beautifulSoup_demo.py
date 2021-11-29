from bs4 import BeautifulSoup

def main():
    # example training data
    raw_data = """<DOC>
<TEXT id="1200405109523201430_2_2x2">
别只能想自己，想你周围的人。还有你，如果你是一个家庭的爸爸，你多想自己的孩子；如果你是青少年你多想自己的未来；那你可以禁烟了。</TEXT>
<CORRECTION>
别只想自己，要想想你周围的人。还有，如果你是一个家庭的爸爸，你要多想想自己的孩子；如果你是青少年你要多想想自己的未来；那你就可以戒烟了。
</CORRECTION>
<ERROR start_off="3" end_off="3" type="R"></ERROR>
<ERROR start_off="8" end_off="8" type="M"></ERROR>
<ERROR start_off="9" end_off="9" type="M"></ERROR>
<ERROR start_off="17" end_off="17" type="R"></ERROR>
<ERROR start_off="32" end_off="32" type="M"></ERROR>
<ERROR start_off="34" end_off="34" type="M"></ERROR>
<ERROR start_off="48" end_off="48" type="M"></ERROR>
<ERROR start_off="50" end_off="50" type="M"></ERROR>
<ERROR start_off="58" end_off="58" type="M"></ERROR>
<ERROR start_off="60" end_off="60" type="S"></ERROR>
</DOC>

<DOC>
<TEXT id="20200209550523150152_2_2x1">
若父母主动和孩子平起平坐，倾心交谈，互相充分理解对方，会能够大幅减少互相抱怨的情况。
</TEXT>
<CORRECTION>
若父母主动和孩子平起平坐，倾心交谈，互相充分理解对方，会大幅减少互相抱怨的情况。
</CORRECTION>
<ERROR start_off="29" end_off="30" type="R"></ERROR>
</DOC>
    """

    # parse raw_data and store the result in soup
    soup = BeautifulSoup(raw_data, 'html.parser')

    # # print what is stored in soup
    # print(soup.prettify())

    # extract data
    docs = soup.find_all('doc')     # find all object with tag 'doc'
    for doc in docs:
        text = doc.find('text').string.strip()
        correction = doc.find('correction').string.strip()
        raw_errors = doc.find_all('error')
        errors = []
        for err in raw_errors:
            tuple = (err['end_off'], err['start_off'], err['type'])
            errors.append(tuple)

        print(f"text: {text}")
        print(f"correction: {correction}")
        print(f"errors: {errors}")
        print()


if __name__ == '__main__':
    main()