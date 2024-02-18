import spacy


def load_spacy_model(model='en_core_web_trf'):
    nlp = spacy.load(model)
    return nlp


def process_sentence(sentence, nlp):
    doc = nlp(sentence)
    sentence_for_spacy = []

    for i, token in enumerate(doc):
        if token.text == ' ':
            continue
        sentence_for_spacy.append(token.text)

    sentence_for_spacy = ' '.join(sentence_for_spacy)
    noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(
        sentence_for_spacy, nlp, need_index=True)
    return noun_phrase


def extract_noun_phrase(text, nlp, need_index=False):
    """
    Extract noun phrase from text. nlp is a spacy model.
    Args:
        text: str, text to be processed.
        nlp: spacy model.
        need_index: bool, whether to return the index of the noun phrase.
    Returns:
        noun_phrase: str, noun phrase of the text.
    """
    # text = text.lower()

    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text
