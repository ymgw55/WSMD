from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.ot_utils import get_conti_token


def BOWS(tokens1, tokens2, tokenizer):

    conti_token = get_conti_token(tokenizer)
    if conti_token == '':
        space_token = 'Ġ'
        res_tokens1 = [tk.replace(space_token, '') for tk in tokens1]
        res_tokens2 = [tk.replace(space_token, '') for tk in tokens2]
    else:
        res_tokens1 = tokens1
        res_tokens2 = tokens2

    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    bow = vectorizer.fit_transform(
        [' '.join(res_tokens1), ' '.join(res_tokens2)])

    emb1 = bow[0].toarray()[0]
    emb2 = bow[1].toarray()[0]

    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

    return sim


def get_sif_weight(
        count_fn='/working/data/datasets/sif/enwiki_vocab_min200.txt',
        a=1e-3):
    word2weight = defaultdict(lambda: 1)
    with open(count_fn) as f:
        lines = f.readlines()
    N = 0
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if len(line.split()) == 2:
                k, v = line.split()
                k = k.lower()
                v = float(v)
                word2weight[k] = v
                N += v
            else:
                print(line)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value / N)
    return word2weight


def get_usif_weight(
        count_fn='/working/data/datasets/sif/enwiki_vocab_min200.txt',
        n=11):

    prob = {}
    total = 0.0

    for line in open(count_fn, encoding='utf8'):
        k, v = line.split()
        v = int(v)
        k = k.lower()

        prob[k] = v
        total += v

    prob = {k: (prob[k] / total) for k in prob}
    min_prob = min(prob.values())

    vocab_size = float(len(prob))
    threshold = 1 - (1 - 1 / vocab_size) ** n
    alpha = len([w for w in prob if prob[w] > threshold]) / vocab_size
    Z = 0.5 * vocab_size
    a = (1 - alpha) / (alpha * Z)

    word2weight = defaultdict(lambda: a / (0.5 * a + min_prob))
    for w in prob:
        word2weight[w] = a / (0.5 * a + prob[w])

    return word2weight


def test(rm_sw=True):
    import string

    import torch
    from nltk.corpus import stopwords
    from transformers import AutoModel, AutoTokenizer

    for model_name in ['bert-base-uncased', 'roberta-base']:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token = tokenizer.pad_token
        space_token = 'Ġ'

        if rm_sw:
            stop_words = stopwords.words('english')
            punct = [i for i in string.punctuation]
            stop_words = stop_words + punct + \
                [cls_token, sep_token, pad_token, space_token]
            stop_words = set(stop_words)
        else:
            stop_words = set([cls_token, sep_token, pad_token, space_token])

        # sample long sentence
        sents = ['The quick brown fox jumps over the lazy dog.',
                'The lazy cat jumps over the quick brown bear.']  # noqa

        tokens = []
        embs = []
        for sent in sents:
            sent = sent.lower()
            tks = tokenizer.tokenize(sent)
            tks = [tokenizer.cls_token] + tks + [tokenizer.sep_token]
            tokens.append(tks)

            emb = model(torch.tensor([tokenizer.convert_tokens_to_ids(tks)])
                        )[0][0]
            emb = emb.detach().cpu().numpy()
            embs.append(emb)
        tokens1, tokens2 = tokens
        emb1, emb2 = embs

        bows = BOWS(tokens1, tokens2, stop_words, tokenizer)
        print(f'BOWS: {bows}')


if __name__ == '__main__':
    test()
