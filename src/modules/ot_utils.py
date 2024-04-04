def stanza2dic(stanza_result):
    dic_doc = []
    for sent in stanza_result.sentences:
        dic_sent = []
        for word in sent.words:
            tmp = {'id': (word.id), 'text': word.text,
                   'upos': word.upos, 'xpos': word.xpos,
                   'head': word.head, 'deps': word.deps,
                   'deprel': word.deprel}
            dic_sent.append(tmp)
        dic_doc.append(dic_sent)
    return dic_doc


def stanza2dic_sent(stanza_result):
    assert len(stanza_result.sentences) == 1
    sent = stanza_result.sentences[0]
    dic_sent = []
    for word in sent.words:
        tmp = {'id': (word.id), 'text': word.text,
               'upos': word.upos, 'xpos': word.xpos,
               'head': word.head, 'deps': word.deps,
               'deprel': word.deprel}
        dic_sent.append(tmp)
    return dic_sent


def subword2word(seq_word, conti_token):
    seq = []
    for w in seq_word:
        if len(w) == 1:
            seq.append(w[0])
        else:
            if conti_token == '##':
                w = [w[0]] + [x[2:] for x in w[1:]]
                seq.append(''.join(w))
            elif conti_token == '':
                seq.append(''.join(w))

    if conti_token == '':
        seq = [x.replace('Ġ', '') for x in seq]

    return seq


def get_conti_token(tokenizer):
    if tokenizer.__class__.__name__ in (
        'BertTokenizer', 'BertTokenizerFast',
        'DistilBertTokenizer', 'DistilBertTokenizerFast',
    ):
        conti_token = '##'
    elif tokenizer.__class__.__name__ in (
            'RobertaTokenizer', 'RobertaTokenizerFast'):
        conti_token = ''
    else:
        print(tokenizer.__class__.__name__)
        raise NotImplementedError
    return conti_token


def tokens2words(tokens, tokenizer):

    spec_tokens = \
        [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]

    conti_token = get_conti_token(tokenizer)

    seq_id = []
    seq_word = []
    for i, tk in enumerate(tokens):
        if tk in spec_tokens:
            continue
        if len(seq_id) == 0:
            seq_id.append([i])
            seq_word.append([tk])
            continue
        if conti_token == '' and not tk.startswith('Ġ'):
            seq_id[-1].append(i)
            seq_word[-1].append(tk)
        elif conti_token == '##' and tk.startswith('##'):
            seq_id[-1].append(i)
            seq_word[-1].append(tk)
        else:
            seq_id.append([i])
            seq_word.append([tk])
    seq_word = subword2word(seq_word, conti_token)
    return seq_id, seq_word


def rm_stopwords(tokens, stop_words, tokenizer):
    conti_token = get_conti_token(tokenizer)
    stop_ids = []
    for k, w in enumerate(tokens):
        if conti_token == '':
            if not w[0] == ('Ġ'):
                if k == 1 and w not in stop_words:
                    continue
                stop_ids.append(k)
            else:
                if w.replace('Ġ', '') in stop_words:
                    stop_ids.append(k)
        elif conti_token == '##':
            if w in stop_words:
                stop_ids.append(k)
            elif w[:2] == '##':
                stop_ids.append(k)

    return stop_ids


def clean_tokens(tokens):
    tokens = [x.replace('Ġ', '') for x in tokens]
    tokens = [x.replace('``', '`') for x in tokens]
    return tokens


def test():
    from transformers import AutoTokenizer
    for model_name in ['bert-base-uncased', 'roberta-base',
                       'princeton-nlp/unsup-simcse-bert-base-uncased',
                       'sentence-transformers/all-distilroberta-v1']:
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sent = 'The city sits at the confluence of the Snake River with the great Weiser River , which marks the border with Oregon .'  # noqa
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + \
            [tokenizer.sep_token]
        seq_id, seq_word = tokens2words(tokens, tokenizer)
        print(seq_id)
        print(seq_word)


if __name__ == '__main__':
    test()
