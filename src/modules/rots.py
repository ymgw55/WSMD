from collections import defaultdict
from time import time

import numpy as np
import ot
import spacy

spacy_nlp = spacy.load('en_core_web_sm')


def fix_words(doc, words):
    cd = 0
    valid_cds = []
    for cw in range(len(words)):
        if cd >= len(doc):
            break
        if words[cw] == doc[cd].text.strip():
            valid_cds.append(cd)
            tmp = 1
        else:
            tmp = 0
            while cd + tmp < len(doc):
                n = len(doc[cd + tmp].text)
                if words[cw][:n] == doc[cd + tmp].text:
                    valid_cds.append(cd)
                    tmp += 1
                elif words[cw][-n:] == doc[cd + tmp].text:
                    tmp += 1
                    break
                else:
                    tmp += 1
        cd += tmp

    tmp_words = []
    for cd in valid_cds:
        tmp_words.append(doc[cd].text)
    print(valid_cds, words, tmp_words, [doc[idx] for idx in range(len(doc))])
    assert len(valid_cds) == len(words)
    return tmp_words


def rm_words(embs, words, ws, doc):
    tmp_embs = []
    tmp_words = []
    tmp_ws = []
    for i in range(len(words)):
        if len(words[i]) == 0:
            idx = len(tmp_words)
            if idx >= len(doc) or doc[idx].text != ' ':
                continue
        tmp_embs.append(embs[i])
        tmp_words.append(words[i])
        tmp_ws.append(ws[i])
    assert len(tmp_words) == len(tmp_embs) == len(tmp_ws) == len(doc)
    return tmp_embs, tmp_words, tmp_ws


def words_check(embs, words, ws):
    string_ = ' '.join(words)
    doc = spacy_nlp(string_)
    if len(words) < len(doc):
        words = fix_words(doc, words)
    elif len(words) > len(doc):
        embs, words, ws = rm_words(embs, words, ws, doc)
    return embs, words, ws


class Sentence:
    def __init__(self, vectors, words, weights, **kwargs):
        self.vectors = vectors
        self.weights = weights
        self.words = words
        assert len(self.words) == len(self.vectors) == len(self.words)
        self.tree_level_index = defaultdict(list)
        self.tree_level_span_flat = defaultdict(list)
        # each span is triple (begin, end, parent index)
        self.span_dict = {}  # sid : span
        self.span_vw_dict = {}  # sid : [v*w, w]
        # store the span of each word, which is the element of the tree
        self.span_begin_index = {}

    @property
    def sentence_vector(self):
        sent_vector = 0
        for v, w in zip(self.vectors, self.weights):
            sent_vector += v * w
        if len(self.vectors) == 0:
            return np.ones((1, 300))
        else:
            return sent_vector

    @property
    def string(self):
        return " ".join(self.words)

    def __len__(self):
        return len(self.weights)

    def _span_register(self, span):
        sid = len(self.span_dict)
        self.span_dict[sid] = span
        return sid

    def span_vw(self, span):
        begin, end, _ = span
        chunk_vw = 0
        chunk_wv = 0
        for i in range(begin, end):
            chunk_vw += self.vectors[i] * self.weights[i]
            chunk_wv += self.weights[i] * np.linalg.norm(self.vectors[i])
        return chunk_vw, chunk_wv

    def span_str(self, span):
        begin, end, _ = span
        return " ".join(self.words[begin: end])

    def parse(self, parser):
        # refresh the tree index
        self.tree_level_index = defaultdict(list)
        self.tree_level_span_flat = defaultdict(list)
        self.span_dict = {}
        self.span_vw_dict = {}
        self.span_begin_index = {}
        t1 = time()
        if parser.lower() in 'dependency':
            self._parse_spacy()
        else:
            self._parse_binary()
        parse_time = time() - t1
        self.update_tree_vw()
        return parse_time

    def _parse_spacy(self):
        doc = spacy_nlp(self.string)
        assert len(doc) == len(self.words)

        roots = [t for t in doc if t.head == t]

        def partition(token, level, sid):
            """
            Args:
                token: the token represented by the span,
                which maintains the connection information of the span
                level: the level of the span
                sid: the sid of parent span
            Return:
                begin: the begin of this token
                end: the end of this token
            """
            # target #1 construct the span for this token
            # initialize a dummy span to secure the sid
            token_span = (None, None, sid)
            begin = token.i
            end = token.i + 1
            token_sid = self._span_register(token_span)
            for child_token in token.children:
                b, e = partition(child_token, level+1, token_sid)
                begin = min(begin, b)
                end = max(end, e)
            token_span = (begin, end, sid)
            self.span_dict[token_sid] = token_span
            self.tree_level_index[level].append(token_span)

            if end - begin > 1:
                word_span = (token.i, token.i+1, token_sid)
                self._span_register(word_span)
                self.tree_level_index[level+1].append(word_span)
            return begin, end

        if len(roots) == 1:
            partition(roots[0], 0, None)
        else:
            root_span = (0, len(self), None)
            rsid = self._span_register(root_span)
            self.tree_level_index[0].append(root_span)
            for token in roots:
                partition(token, 1, rsid)

        for l in self.tree_level_index:
            self.tree_level_index[l] = sorted(
                self.tree_level_index[l], key=lambda x: x[0])

    def _parse_binary(self):
        def partition(span, level, sid):
            begin, end, _ = span
            # if can be further divided
            if end - begin > 1:
                mid = (begin + end) // 2
                assert begin < mid < end

                sub_span1 = (begin, mid, sid)
                ssid1 = self._span_register(sub_span1)
                self.tree_level_index[level].append(sub_span1)
                partition(sub_span1, level + 1, ssid1)

                sub_span2 = (mid, end, sid)
                ssid2 = self._span_register(sub_span2)
                self.tree_level_index[level].append(sub_span2)
                partition(sub_span2, level + 1, ssid2)

        root_span = (0, len(self), None)
        rsid = self._span_register(root_span)
        self.tree_level_index[0] = [root_span]
        partition(root_span, 1, rsid)

    def update_tree_vw(self):
        if len(self.span_dict) == 0:
            return
        d = len(self.tree_level_index) - 1
        while d >= 0:
            for span in self.tree_level_index[d]:
                b, e, p = span
                if e - b == 1:  # if is single word
                    vw, wvnorm = self.span_vw(span)
                    self.span_vw_dict[span] = [vw, wvnorm]
                    self.span_begin_index[b] = span
                else:
                    vw, wvnorm = self.span_vw_dict[span]

                if p in self.span_dict:
                    p_span = self.span_dict[p]
                    if p_span in self.span_vw_dict:
                        _vw, _wvnorm = self.span_vw_dict[p_span]
                        self.span_vw_dict[p_span] = [
                            _vw + vw, _wvnorm + wvnorm]
                    else:
                        self.span_vw_dict[p_span] = [vw, wvnorm]
            d -= 1
        return

    def get_level_vectors_weights(self, l):
        # l = min(len(self.tree_level_index)-1, l)
        vectors, wvnorms = [], []
        b = 0
        li = 0
        while b < len(self):
            if li < len(self.tree_level_index[l]) and \
                    b == self.tree_level_index[l][li][0]:
                get_level_span = self.tree_level_index[l][li]
                _vw, _wvnorm = self.span_vw_dict[get_level_span]
                vectors.append(_vw)
                wvnorms.append(_wvnorm)
                b = get_level_span[1]
                li += 1
                self.tree_level_span_flat[l].append(get_level_span)
            else:
                get_word_span = self.span_begin_index[b]
                _vw, _wvnorm = self.span_vw_dict[get_word_span]
                vectors.append(_vw)
                wvnorms.append(_wvnorm)
                b += 1
                self.tree_level_span_flat[l].append(get_word_span)

        if (l-1) in self.tree_level_span_flat:
            top_down_link = defaultdict(list)
            down_span_flat = self.tree_level_span_flat[l]
            j = 0
            for i, tsp in enumerate(self.tree_level_span_flat[l-1]):
                tb, te, _ = tsp
                db, de, _ = down_span_flat[j]
                while tb <= db and de <= te:
                    top_down_link[i].append(j)
                    j += 1
                    if j >= len(down_span_flat):
                        break
                    db, de, _ = down_span_flat[j]
        else:
            top_down_link = {}

        return vectors, wvnorms, top_down_link


def cosine(x, y):
    def shape_norm(v):
        v = np.asarray(v)
        if len(v.shape) == 1:
            v = v.reshape(1, -1)
        return v
    x, y = shape_norm(x), shape_norm(y)
    n1, d1 = x.shape
    n2, d2 = y.shape
    assert d1 == d2
    M = 1 - (np.dot(x, y.T)) / (np.linalg.norm(x, axis=1, keepdims=True)
                                * np.linalg.norm(y, axis=1,
                                                 keepdims=True).T + 1e-10)
    return np.squeeze(M).reshape(n1, n2)


class CosineSimilarity:
    def __init__(self, adjust_WRD=False, **kwargs):
        self.adjust_WRD = adjust_WRD

    def __call__(self, s1, s2):
        sim = 1 - cosine(s1.sentence_vector, s2.sentence_vector).squeeze()
        if self.adjust_WRD:
            sim *= np.linalg.norm(s1.sentence_vector)
            sim *= np.linalg.norm(s2.sentence_vector)
            a = np.asarray([np.linalg.norm(v) * w for v,
                           w in zip(s1.vectors, s1.weights)])
            sim /= np.sum(a)
            b = np.asarray([np.linalg.norm(v) * w for v,
                           w in zip(s2.vectors, s2.weights)])
            sim /= np.sum(b)
        return sim


class ROTS:
    def __init__(self, parser, aggregation, margin,
                 depth, preg, creg, ereg, coef_C):
        """
        Args:
            parser: type of parsers, in ['dependency', 'binary']
            depth: how deep you consider this
            preg: prior regularization
            creg: cosine regularization
            ereg: entropy regularization
            coef_C: C interpolation coefficient
            aggregation: 
            how to handle different scores [mean, max, min, last, no]
        """
        self.parser = parser
        self.depth = depth
        if isinstance(preg, list):
            self.prior_reg = preg
        else:
            self.prior_reg = [preg * (i+1) for i in range(depth)]
            # self.prior_reg = [32 for i in range(depth)]
        self.creg = creg
        self.ereg = ereg
        self.coef_C = coef_C
        self.aggregation = aggregation
        self.margin = margin

    def __call__(self, s1: Sentence, s2: Sentence):
        if len(s1.vectors) == 0 or len(s2.vectors) == 0:
            _depth = self.depth
            answer = {d: 1 for d in range(self.depth)}
        elif len(s1.vectors) == 1 or len(s2.vectors) == 1:
            answer = {d: float(CosineSimilarity()(s1, s2))
                      for d in range(self.depth)}
            _depth = self.depth
        else:
            s1.parse(self.parser)
            s2.parse(self.parser)
            _depth = min(max(len(s1.tree_level_index),
                         len(s2.tree_level_index)), self.depth)

            answer = {}  # d, alignment score
            transport_plan = {}
            for d in range(self.depth):
                vectors1, wvnorms1, tdlink1 = s1.get_level_vectors_weights(d)
                vectors2, wvnorms2, tdlink2 = s2.get_level_vectors_weights(d)
                M_cossim = cosine(vectors1, vectors2)
                if self.margin == 'norm_vectors':
                    _a = np.asarray([np.linalg.norm(v) for v in vectors1])
                    _b = np.asarray([np.linalg.norm(v) for v in vectors2])
                elif self.margin == 'vector_norms':
                    _a = np.asarray(wvnorms1)
                    _b = np.asarray(wvnorms2)
                else:
                    raise NotImplementedError
                a = _a / np.sum(_a)
                b = _b / np.sum(_b)
                C = np.sum(_a) * np.sum(_b) / (
                    np.linalg.norm(s1.sentence_vector)
                    * np.linalg.norm(s2.sentence_vector) + 1e-3)
                cos_prior = a.reshape(-1, 1).dot(b.reshape(1, -1))
                if tdlink1 and tdlink2:
                    prior_plan_top = transport_plan[d-1]
                    prior_plan_down = np.copy(cos_prior)
                    for ti in tdlink1:
                        for tj in tdlink2:
                            mass = prior_plan_top[ti, tj]
                            local_a = np.sum([_a[di] for di in tdlink1[ti]])
                            local_b = np.sum([_b[dj] for dj in tdlink2[tj]])
                            for di in tdlink1[ti]:
                                for dj in tdlink2[tj]:
                                    prior_plan_down[di, dj] = mass * \
                                        _a[di] * _b[dj] / local_a / local_b
                else:
                    prior_plan_down = cos_prior
                M = M_cossim - np.log(prior_plan_down +
                                      1e-10) * self.prior_reg[d]
                reg = self.creg + self.prior_reg[d] + self.ereg
                if d == 0:
                    P = ot.emd(a, b, M)
                else:
                    P = ot.sinkhorn(a, b, M, reg, method='sinkhorn_stabilized')
                # P = ot.emd(a, b, M)
                # P = ot.unbalanced.sinkhorn_unbalanced(
                #   a, b, M, reg=reg, reg_m=1, method="sinkhorn_stabilized")
                transport_plan[d] = P
                answer[d] = (1 - np.sum(P * M_cossim)) * \
                    (1 - self.coef_C + self.coef_C * C)
                # answer[d] = 1 - ot.emd2(a, b, M)

        if self.aggregation == 'mean':
            return np.mean(list(answer.values()))
        elif self.aggregation == 'max':
            return np.max(list(answer.values()))
        elif self.aggregation == 'min':
            return np.min(list(answer.values()))
        elif self.aggregation == 'last':
            return answer[_depth-1]
        elif self.aggregation == 'all':
            answer['mean'] = np.mean(list(answer.values()))
            answer['max'] = np.max(list(answer.values()))
            answer['min'] = np.min(list(answer.values()))
            answer['last'] = answer[_depth-1]
            return answer
        else:
            return answer


def test():
    words = ['twin', 'falls', 'part', 'jerome', 'county',
             'id', 'micro', 'statistical' 'area']
    doc = spacy_nlp(' '.join(words))
    print(doc)
    if len(words) < len(doc):
        print(fix_words(doc, words))


if __name__ == '__main__':
    test()
