def _doc_to_text(doc, cot=False):
    prepend = "Fill in the blank: "
    text = prepend + f"prepend{doc['sentence']}\n(A) {doc['option1']}\n(B) {doc['option2']}\n"
    if cot:
        text += "Let's think step by step."
    return text

def doc_to_text(doc):
    return _doc_to_text(doc, cot=False)

def doc_to_text_cot(doc):
    return _doc_to_text(doc, cot=True)

def doc_to_target(doc):
    choices = ['(A)', '(B)']
    return choices[doc['option2'] == doc['answer']]
