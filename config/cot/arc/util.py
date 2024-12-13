def _doc_to_text(doc, cot=False):
    text = doc['question'] + '\nChoose the best continuation:\n'

    letters = ['A', 'B', 'C', 'D', 'E']
    choices = doc['choices']['text']
    choice_strs = [
        f'\n({letters[i]}) {choice}' for i, choice in enumerate(choices)
        ]

    text += ''.join(choice_strs)

    text += '\nA: '
    if cot:
        text += "\nLet's think step by step: "
    return text


def doc_to_text_cot(doc):
    return _doc_to_text(doc, cot=True)

def doc_to_text_gen(doc):
    return _doc_to_text(doc, cot=False)
