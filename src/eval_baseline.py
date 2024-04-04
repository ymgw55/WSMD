import pickle
import string
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot
import pandas as pd
import stanza
import yaml
from nltk.corpus import stopwords
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.baseline import BOWS, get_sif_weight, get_usif_weight
from modules.ot_distances import (get_WMD_cost, get_WRD_cost,
                                  matrices_elemnt_mean, matrix_elemnt_mean)
from modules.ot_utils import (clean_tokens, rm_stopwords, stanza2dic_sent,
                              tokens2words)
from modules.owmd import sinkhorn_knopp
from modules.rots import ROTS, Sentence, words_check
from modules.synwmd import get_SWD_cost
from modules.wmdo import get_penalty
from utils import get_logger

warnings.filterwarnings("ignore")

stanza.download('en')
nlp = stanza.Pipeline(lang='en',
                      processors='tokenize,mwt,pos,lemma,depparse',
                      tokenize_pretokenized=True,
                      tokenize_no_ssplit=True,
                      verbose=True,
                      use_gpu=True)


def bert_score(sent1_embs, sent2_embs, w1, w2):
    # input
    # sent1_embs: (n1, 768)
    # sent2_embs: (n2, 768)
    # w1: (n1,)
    # w2: (n2,)

    # recall
    r = 0
    for i in range(len(sent1_embs)):
        x_i = sent1_embs[i]
        max_dot = np.max(np.dot(x_i.reshape(1, -1), sent2_embs.T))
        r += max_dot * w1[i]
    r = r / np.sum(w1)

    # precision
    p = 0
    for j in range(len(sent2_embs)):
        y_j = sent2_embs[j]
        max_dot = np.max(np.dot(y_j.reshape(1, -1), sent1_embs.T))
        p += max_dot * w2[j]
    p = p / np.sum(w2)

    # f1
    f1 = 2 * r * p / (r + p + 1e-30)
    return f1


def dynamax(sent1_embs, sent2_embs):
    # input
    # sent1_embs: (n1, 768)
    # sent2_embs: (n2, 768)

    # concat
    u = np.concatenate([sent1_embs, sent2_embs], axis=0)  # (n1+n2, 768)
    max_1 = np.max(np.dot(sent1_embs, u.T), axis=0)  # (n1+n2,)
    z = np.zeros_like(max_1)
    max_1 = np.maximum(max_1, z)

    max_2 = np.max(np.dot(sent2_embs, u.T), axis=0)  # (n1+n2,)
    z = np.zeros_like(max_2)
    max_2 = np.maximum(max_2, z)

    r = np.minimum(max_1, max_2)  # (n1+n2,)
    q = np.maximum(max_1, max_2)  # (n1+n2,)
    dmj = np.sum(r) / (np.sum(q) + 1e-30)

    return dmj


def baseline_method(
        method, rm_sw, weight_name, params,
        task, dataset, model_name, emb_layer,
        whiten, normalize, full, exp_dir, root_dir, module=None):

    log_file = str(exp_dir / 'output.log')
    logger = get_logger(log_file=log_file, stream=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        if method == 'cls':
            stop_words = set([sep_token, pad_token, space_token])
        else:
            stop_words = set([cls_token, sep_token, pad_token, space_token])

    if task == 'STSB':
        from datasets.sts import STSB, data_all_set
        task_func = STSB
    elif task == 'SICK':
        from datasets.sick import SICK, data_all_set
        task_func = SICK
    elif task in ('STS12', 'STS13', 'STS14', 'STS15', 'STS16'):
        from datasets.sts12_16 import (STS12, STS13, STS14, STS15, STS16,
                                       data_all_set)
        task_func = eval(task)
    elif task == 'PAWSWiki':
        if full:
            from datasets.paws import PAWSWiki
            from datasets.paws import data_all_full_set as data_all_set
        else:
            from datasets.paws import PAWSWiki, data_all_set
        task_func = PAWSWiki
    elif task == 'PAWSQQP':
        if full:
            from datasets.paws import PAWSQQP
            from datasets.paws import data_all_full_set as data_all_set
        else:
            from datasets.paws import PAWSQQP, data_all_set
        task_func = PAWSQQP
    else:
        raise ValueError(f'Invalid task: {task}')

    _, _, gs_scores, _ = \
        data_all_set(task, task_func, dataset)

    if weight_name == 'sif':
        word2weight = get_sif_weight()
    elif weight_name == 'usif':
        word2weight = get_usif_weight()
    else:
        word2weight = defaultdict(lambda: 1.0)

    full_str = 'full_' if full else ''
    whiten_str = 'whiten_' if whiten else ''
    normalize_str = 'normalize_' if normalize else ''
    dir_name = f'{full_str}{whiten_str}{normalize_str}model'
    model_dir = root_dir /\
        f'data/{dir_name}/{task}/{dataset}/{model_name.replace("/", "_")}'
    logger.info(f'model_dir: {model_dir}')

    with open(model_dir / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    sent1s_results, sent2s_results = results

    if method in ('conneg', 'rots'):
        vocab_matrix = []
        for (sent1_embs, _, _,
                sent2_embs, _, _) in tqdm(list(
                zip(*sent1s_results, *sent2s_results))):
            sent1_emb = sent1_embs[emb_layer]
            sent2_emb = sent2_embs[emb_layer]
            for emb in sent1_emb:
                vocab_matrix.append(emb)
            for emb in sent2_emb:
                vocab_matrix.append(emb)
        vocab_matrix = np.array(vocab_matrix)

        # conceptor negation
        n, d = vocab_matrix.shape
        R = 0
        for i in range(n):
            w = vocab_matrix[i].reshape(-1, 1)
            R += np.dot(w, w.T)
        R /= n
        C = np.dot(R, np.linalg.inv(R + np.eye(d) * params['alpha'] ** -2))
        A = np.eye(d) - C

        def transfer(x):
            return np.dot(A, x.reshape(-1, 1)).reshape(-1)

        tmp_vocab_matrix = np.zeros_like(vocab_matrix)
        for i in range(n):
            tmp_vocab_matrix[i] = transfer(vocab_matrix[i])
        vocab_matrix = tmp_vocab_matrix

        # scaling
        div_vec = np.linalg.norm(np.array(vocab_matrix), axis=0) + 1e-30
    else:
        def transfer(x):
            return x
        div_vec = np.ones(768)

    valid_sent1_embs = []
    valid_sent2_embs = []
    valid_sent1_tokens = []
    valid_sent2_tokens = []
    valid_w1s = []
    valid_w2s = []
    sent_vecs = []
    for (sent1_embs, _, sent1_tokens,
            sent2_embs, _, sent2_tokens) in tqdm(list(
            zip(*sent1s_results, *sent2s_results))):
        sent1_t2w, sent1_word_seq = tokens2words(sent1_tokens, tokenizer)
        sent2_t2w, sent2_word_seq = tokens2words(sent2_tokens, tokenizer)
        sent1_parsing_data = nlp('\n\n'.join([' '.join(sent1_word_seq)]))
        sent1_parsing_data = stanza2dic_sent(sent1_parsing_data)
        sent2_parsing_data = nlp('\n\n'.join([' '.join(sent2_word_seq)]))
        sent2_parsing_data = stanza2dic_sent(sent2_parsing_data)
        assert len(sent1_parsing_data) == len(sent1_t2w)
        assert len(sent2_parsing_data) == len(sent2_t2w)

        w1 = np.zeros(len(sent1_tokens))
        for w_id, w in enumerate(sent1_parsing_data):
            w1[sent1_t2w[w_id]] = word2weight[w['text']]
        w2 = np.zeros(len(sent2_tokens))
        for w_id, w in enumerate(sent2_parsing_data):
            w2[sent2_t2w[w_id]] = word2weight[w['text']]

        stop_ids1 = rm_stopwords(sent1_tokens, stop_words, tokenizer)
        sent1_emb = np.delete(sent1_embs[emb_layer], stop_ids1, axis=0)
        sent1_tokens = np.delete(sent1_tokens, stop_ids1)
        w1 = np.delete(w1, stop_ids1)

        if len(sent1_emb) == 0:
            sent1_emb = np.zeros((1, 768))
            sent1_tokens = np.array([''])
            w1 = np.array([0])

        w1 = w1 / len(w1)
        valid_sent1_embs.append(sent1_emb)
        valid_sent1_tokens.append(sent1_tokens)
        valid_w1s.append(w1)

        stop_ids2 = rm_stopwords(sent2_tokens, stop_words, tokenizer)
        sent2_emb = np.delete(sent2_embs[emb_layer], stop_ids2, axis=0)
        sent2_tokens = np.delete(sent2_tokens, stop_ids2)
        w2 = np.delete(w2, stop_ids2)

        if len(sent2_emb) == 0:
            sent2_emb = np.zeros((1, 768))
            sent2_tokens = np.array([''])
            w2 = np.array([0])

        w2 = w2 / len(w2)
        valid_sent2_embs.append(sent2_emb)
        valid_sent2_tokens.append(sent2_tokens)
        valid_w2s.append(w2)

        sum1_emb = 0
        for i in range(len(sent1_emb)):
            emb = sent1_emb[i]
            emb = transfer(emb) / div_vec
            sum1_emb += emb * w1[i]
        sent_vecs.append(sum1_emb)

        sum2_emb = 0
        for i in range(len(sent2_emb)):
            emb = sent2_emb[i]
            emb = transfer(emb) / div_vec
            sum2_emb += emb * w2[i]
        sent_vecs.append(sum2_emb)

    sent_vecs = np.array(sent_vecs)

    if method in ('sif', 'usif'):
        from sklearn.decomposition import TruncatedSVD

        def proj(x, pc):
            return x.dot(pc.transpose()) * pc

        assert params['n_comp'] > 0
        svd = TruncatedSVD(n_components=params['n_comp'])
        svd.fit(sent_vecs)

        lambda_i = []
        p_comp_i = []
        for i in range(params['n_comp']):
            lambda_i.append((svd.singular_values_[i] ** 2
                             ) / np.sum(svd.singular_values_ ** 2))
            p_comp_i.append(svd.components_[i])

        tmp_valid_sent1_embs = []
        for vectors, weights in zip(valid_sent1_embs, valid_w1s):
            sum_weights = sum(weights)
            for i in range(params['n_comp']):
                vectors = [v - proj(v, p_comp_i[i]) * lambda_i[i] * w /
                           sum_weights for v, w in zip(vectors, weights)]
            tmp_valid_sent1_embs.append(vectors)
        valid_sent1_embs = tmp_valid_sent1_embs

        tmp_valid_sent2_embs = []
        for vectors, weights in zip(valid_sent2_embs, valid_w2s):
            sum_weights = sum(weights)
            for i in range(params['n_comp']):
                vectors = [v - proj(v, p_comp_i[i]) * lambda_i[i] * w /
                           sum_weights for v, w in zip(vectors, weights)]
            tmp_valid_sent2_embs.append(vectors)
        valid_sent2_embs = tmp_valid_sent2_embs

    sims = []
    for (sent1_embs, sent1_tokens, w1,
         sent2_embs, sent2_tokens, w2) in tqdm(list(
            zip(valid_sent1_embs, valid_sent1_tokens, valid_w1s,
                valid_sent2_embs, valid_sent2_tokens, valid_w2s))):

        if sum(w1) == 0 or sum(w2) == 0:
            sims.append(0)
            continue

        if method == 'bows':
            sim = BOWS(sent1_tokens, sent2_tokens, tokenizer)
        elif method == 'rots':
            sent1_embs, words1, w1 = words_check(
                sent1_embs, clean_tokens(sent1_tokens), w1)
            sent2_embs, words2, w2 = words_check(
                sent2_embs, clean_tokens(sent2_tokens), w2)
            sim = module(
                Sentence(sent1_embs, words1, w1),
                Sentence(sent2_embs, words2, w2))
        elif method == 'bertscore':
            sim = bert_score(sent1_embs, sent2_embs, w1, w2)
        elif method == 'dynamax':
            sim = dynamax(sent1_embs, sent2_embs)
        elif method == 'cls':
            sent1_cls = sent1_embs[0]
            sent2_cls = sent2_embs[0]
            assert sent1_cls.shape == sent2_cls.shape == (768,)
            sim = cosine_similarity(sent1_cls.reshape(1, -1),
                                    sent2_cls.reshape(1, -1))[0][0]
        else:
            sent1_vec = 0
            for emb, w in zip(sent1_embs, w1):
                sent1_vec += emb * w
            sent2_vec = 0
            for emb, w in zip(sent2_embs, w2):
                sent2_vec += emb * w
            sim = cosine_similarity(sent1_vec.reshape(1, -1),
                                    sent2_vec.reshape(1, -1))[0][0]
        sims.append(sim)

    # replace -inf with min value
    sims = np.array(sims)
    gs_scores = np.array(gs_scores)

    # save tmp and gs_scores
    with open(exp_dir / 'sims.pkl', 'wb') as f:
        pickle.dump(sims, f)
    with open(exp_dir / 'gs_scores.pkl', 'wb') as f:
        pickle.dump(gs_scores, f)

    res_dict = {
        'method': method,
        'rm_sw': rm_sw,
        'wsmd': False,
        'lambd': '',
        'weight_name': weight_name,
        'cost_name': '',
        'params': params,
        'model_name': model_name,
        'emb_layer': emb_layer,
        'attn_layer': '',
        'whiten': whiten,
        'normalize': normalize,
        'task': task,
        'dataset': dataset,
    }

    if task in ('STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK', 'STSB'):
        pearson = pearsonr(sims, gs_scores)[0]
        spearman = spearmanr(sims, gs_scores)[0]
        res_dict['pearson'] = pearson
        res_dict['spearman'] = spearman
    else:
        auc = roc_auc_score(gs_scores, sims)
        res_dict['auc'] = auc

    save_path = str(exp_dir / 'result.csv')
    df = pd.DataFrame([res_dict])
    df.to_csv(save_path, index=False)


def sim_calculate(
        method,
        sent1_emb, sent1_sam, sent1_tokens,
        sent2_emb, sent2_sam, sent2_tokens,
        wsmd, word2weight, cost_name, params, stop_words, tokenizer, lambd):

    sent1_emb = sent1_emb.copy()
    sent2_emb = sent2_emb.copy()

    if cost_name == 'swd':
        # normaize emb
        sent1_emb = sent1_emb / np.linalg.norm(
            sent1_emb, axis=1, keepdims=True)
        sent2_emb = sent2_emb / np.linalg.norm(
            sent2_emb, axis=1, keepdims=True)

    sent1_t2w, sent1_word_seq = tokens2words(sent1_tokens, tokenizer)
    sent2_t2w, sent2_word_seq = tokens2words(sent2_tokens, tokenizer)
    sent1_parsing_data = nlp('\n\n'.join([' '.join(sent1_word_seq)]))
    sent1_parsing_data = stanza2dic_sent(sent1_parsing_data)
    sent2_parsing_data = nlp('\n\n'.join([' '.join(sent2_word_seq)]))
    sent2_parsing_data = stanza2dic_sent(sent2_parsing_data)
    assert len(sent1_parsing_data) == len(sent1_t2w)
    assert len(sent2_parsing_data) == len(sent2_t2w)

    # assign weight
    if word2weight is None:
        # norm weight
        w1 = np.zeros(len(sent1_tokens))
        assert len(sent1_emb) == len(w1)
        for w_id, w in enumerate(sent1_parsing_data):
            for t_id in sent1_t2w[w_id]:
                w1[t_id] = np.linalg.norm(sent1_emb[t_id])
        w2 = np.zeros(len(sent2_tokens))
        assert len(sent2_emb) == len(w2)
        for w_id, w in enumerate(sent2_parsing_data):
            for t_id in sent2_t2w[w_id]:
                w2[t_id] = np.linalg.norm(sent2_emb[t_id])
    else:
        min_v = min(word2weight.values())
        w1 = np.zeros(len(sent1_tokens))
        for w_id, w in enumerate(sent1_parsing_data):
            try:
                w1[sent1_t2w[w_id]] = word2weight[w['text']]
            except KeyError:
                w1[sent1_t2w[w_id]] = min_v
        w2 = np.zeros(len(sent2_tokens))
        for w_id, w in enumerate(sent2_parsing_data):
            try:
                w2[sent2_t2w[w_id]] = word2weight[w['text']]
            except KeyError:
                w2[sent2_t2w[w_id]] = min_v

    # rm stop words
    stop_ids1 = rm_stopwords(sent1_tokens, stop_words, tokenizer)
    sent1_emb[stop_ids1] = 0
    w1[stop_ids1] = 0
    stop_ids2 = rm_stopwords(sent2_tokens, stop_words, tokenizer)
    sent2_emb[stop_ids2] = 0
    w2[stop_ids2] = 0

    if np.sum(w1) == 0 or np.sum(w2) == 0:
        return -np.inf

    # weight norm
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)

    # smd doesn't need cost matrix
    if method == 'smd':
        return -ot.gromov.gromov_wasserstein2(
            sent1_sam, sent2_sam, w1, w2, loss_fun='square_loss')

    # cost
    if cost_name == 'euclid':
        C = get_WMD_cost(sent1_emb, sent2_emb)
    elif cost_name == 'cosine':
        C = get_WRD_cost(sent1_emb, sent2_emb)
    elif cost_name == 'swd':
        C = get_SWD_cost(
            sent1_emb, sent1_parsing_data, sent1_t2w, w1,
            sent2_emb, sent2_parsing_data, sent2_t2w, w2, a=params['a'])

    if method in ('wmdo'):
        use_P = True
    else:
        use_P = False

    if method == 'owmd':
        P = sinkhorn_knopp(w1, w2, C, l1=params['l1'], l2=params['l2'],
                           sigma=params['sigma'], max_iter=params['max_iter'])
        otd = np.sum(C * P)
    else:
        if wsmd:
            assert lambd >= 0 and lambd <= 1

            Cm = matrix_elemnt_mean(C)
            Am = matrices_elemnt_mean(sent1_sam, sent2_sam)
            a1 = np.sqrt(Cm / Am) * sent1_sam
            a2 = np.sqrt(Cm / Am) * sent2_sam

            otd = ot.gromov.fused_gromov_wasserstein2(
                C, a1, a2, w1, w2, alpha=lambd, log=False)
            if use_P:
                P = ot.gromov.fused_gromov_wasserstein(
                    C, a1, a2, w1, w2, alpha=lambd, log=False)
        else:
            otd = ot.emd2(w1, w2, C)
            if use_P:
                P = ot.emd(w1, w2, C)

    if method == 'wmdo':
        penalty = get_penalty(sent1_tokens, sent2_tokens, P)
        return -(otd - params['delta'] * (1/2 - penalty))
    else:
        return -otd


def ot_method(method, rm_sw, weight_name, cost_name, params,
              task, dataset, model_name, emb_layer,
              whiten, normalize, full, lambd, exp_dir, root_dir, wsmd=False):

    log_file = str(exp_dir / 'output.log')
    logger = get_logger(log_file=log_file, stream=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    if task == 'STSB':
        from datasets.sts import STSB, data_all_set
        task_func = STSB
    elif task == 'SICK':
        from datasets.sick import SICK, data_all_set
        task_func = SICK
    elif task in ('STS12', 'STS13', 'STS14', 'STS15', 'STS16'):
        from datasets.sts12_16 import (STS12, STS13, STS14, STS15, STS16,
                                       data_all_set)
        task_func = eval(task)
    elif task == 'PAWSWiki':
        if full:
            from datasets.paws import PAWSWiki
            from datasets.paws import data_all_full_set as data_all_set
        else:
            from datasets.paws import PAWSWiki, data_all_set
        task_func = PAWSWiki
    elif task == 'PAWSQQP':
        if full:
            from datasets.paws import PAWSQQP
            from datasets.paws import data_all_full_set as data_all_set
        else:
            from datasets.paws import PAWSQQP, data_all_set
        task_func = PAWSQQP
    else:
        raise ValueError(f'Invalid task: {task}')

    sent1s, sent2s, gs_scores, vocab_count = \
        data_all_set(task, task_func, dataset)

    vocab = [w[0] for w in vocab_count.most_common()]

    if weight_name == 'uniform':
        word2weight = {w: 1 for w in vocab}
    elif weight_name == 'idf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(use_idf=True)
        tf.fit_transform(sent1s + sent2s)
        word2weight = dict(zip(tf.get_feature_names(), tf.idf_))
    elif weight_name == 'swf':
        from modules.synwmd import get_swf_weight
        word2id = {w: id for id, w in enumerate(vocab)}
        words = sent1s + sent2s
        word2weight = get_swf_weight(word2id, words, vocab, nlp)
    elif weight_name == 'norm':
        word2weight = None

    full_str = 'full_' if full else ''
    whiten_str = 'whiten_' if whiten else ''
    normalize_str = 'normalize_' if normalize else ''
    dir_name = f'{full_str}{whiten_str}{normalize_str}model'
    model_dir = root_dir /\
        f'data/{dir_name}/{task}/{dataset}/{model_name.replace("/", "_")}'
    logger.info(f'model_dir: {model_dir}')

    with open(model_dir / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    sent1s_results, sent2s_results = results

    sims = []
    for (sent1_embs, sent1_attns, sent1_tokens,
            sent2_embs, sent2_attns, sent2_tokens) in tqdm(list(
            zip(*sent1s_results, *sent2s_results))):

        sent1_emb = sent1_embs[emb_layer]
        sent2_emb = sent2_embs[emb_layer]

        sent1_sam = sent1_attns[0][0]
        sent2_sam = sent2_attns[0][0]

        sim = sim_calculate(
            method,
            sent1_emb, sent1_sam, sent1_tokens,
            sent2_emb, sent2_sam, sent2_tokens,
            wsmd=wsmd, word2weight=word2weight, cost_name=cost_name,
            params=params, stop_words=stop_words, tokenizer=tokenizer,
            lambd=lambd)
        sims.append(sim)

    # replace -inf with min value
    sims = np.array(sims)
    gs_scores = np.array(gs_scores)
    logger.info(sum(np.isfinite(sims)))

    # save tmp and gs_scores
    with open(exp_dir / 'sims.pkl', 'wb') as f:
        pickle.dump(sims, f)
    with open(exp_dir / 'gs_scores.pkl', 'wb') as f:
        pickle.dump(gs_scores, f)

    res_dict = {
        'method': method,
        'rm_sw': rm_sw,
        'wsmd': wsmd,
        'lambd': lambd,
        'weight_name': weight_name,
        'cost_name': cost_name,
        'params': params,
        'model_name': model_name,
        'emb_layer': emb_layer,
        'attn_layer': '',
        'whiten': whiten,
        'normalize': normalize,
        'task': task,
        'dataset': dataset,
    }

    if task in ('STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK', 'STSB'):
        pearson = pearsonr(sims, gs_scores)[0]
        spearman = spearmanr(sims, gs_scores)[0]
        res_dict['pearson'] = pearson
        res_dict['spearman'] = spearman
    else:
        auc = roc_auc_score(gs_scores, sims)
        res_dict['auc'] = auc

    save_path = str(exp_dir / 'result.csv')
    logger.info(res_dict)
    df = pd.DataFrame([res_dict])
    df.to_csv(save_path, index=False)


def main(model_name='bert-base-uncased', emb_layer=None,
         task='PAWSWiki', dataset='sample',
         whiten=True, normalize=False, full=False, lambd=0.5,
         root_dir='/working'):

    root_dir = Path(root_dir)

    if full:
        output_dir = root_dir / f'output/full_eval_baseline/{task}/{dataset}/'
    else:
        output_dir = root_dir / f'output/eval_baseline/{task}/{dataset}/'
    output_dir.mkdir(parents=True, exist_ok=True)

    assert model_name in ('bert-base-uncased',
                          'roberta-base',
                          'princeton-nlp/unsup-simcse-bert-base-uncased',
                          'distilbert-base-uncased',
                          'google/multiberts-seed_0')

    # BOWS
    if emb_layer is None:
        emb_layer = 0
        output_dir = output_dir / model_name.replace("/", "_")
        b_method = 'bows'
        config_path = root_dir / f'src/configs/{b_method}.yaml'
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        rm_sw = cfg['use_stopwords']
        weight_name = None
        params = None
        exp = b_method
        exp_dir = output_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        baseline_method(
            b_method, rm_sw, weight_name, params,
            task, dataset, model_name, emb_layer,
            whiten, normalize, full, exp_dir, root_dir)
        return

    output_dir = output_dir / model_name.replace("/", "_") \
        / f'emb_layer{emb_layer}'
    
    data = []

    # baseline methods
    for b_method in ['cosine', 'sif', 'usif', 'conneg', 'rots',
                     'bertscore', 'dynamax', 'cls']:

        if b_method in ['cosine', 'sif', 'usif', 'conneg', 'rots']:
            config_path = root_dir / f'src/configs/{b_method}.yaml'
            cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
            rm_sw = cfg['use_stopwords']
            weight_names = cfg['weight_name']
            params = cfg['params']
        else:
            if b_method in ['bertscore', 'dynamax']:
                rm_sw = True
            else:
                rm_sw = False
            weight_names = [None]
            params = None

        for weight_name in weight_names:
            
            if weight_name is None:
                exp = b_method
            else:
                exp = f'{b_method}_{weight_name}'

            exp_dir = output_dir / exp
            exp_dir.mkdir(parents=True, exist_ok=True)
            if (exp_dir / 'result.csv').exists():
                df = pd.read_csv(exp_dir / 'result.csv')
                res = df.iloc[0].to_dict()
                print(res)
                data.append(res)
                continue
            print(exp)

            if b_method == 'rots':
                module = ROTS(**params['rots'])
            else:
                module = None

            res = baseline_method(
                b_method, rm_sw, weight_name, params,
                task, dataset, model_name, emb_layer,
                whiten, normalize, full, exp_dir, root_dir, module)
            data.append(res)

    # WMD-like methods
    for method in ['opwd', 'wmd', 'wrd', 'synwmd', 'wmdo']:
        config_path = root_dir / f'src/configs/{method}.yaml'
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        rm_sw = cfg['use_stopwords']
        weight_names = cfg['weight_name']
        cost_names = cfg['cost_name']
        params = cfg['params']

        for cost_name in cost_names:
            for weight_name in weight_names:
                if method == 'smd' and weight_name in ('uniform', 'idf'):
                    continue
                for wsmd in [False]:
                    if wsmd and method in ('opwd', 'smd'):
                        continue
                    if wsmd:
                        exp = f'{method}_wsmd_{weight_name}'
                    else:
                        exp = f'{method}_{weight_name}'
                    if cost_name is not None:
                        exp += f'_{cost_name}'

                    exp_dir = output_dir / exp
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    if (exp_dir / 'result.csv').exists():
                        df = pd.read_csv(exp_dir / 'result.csv')
                        res = df.iloc[0].to_dict()
                        print(res)
                        data.append(res)
                        continue
                    print(exp)

                    res = ot_method(method, rm_sw, weight_name, cost_name,
                        params, task, dataset, model_name, emb_layer, whiten,
                        normalize, full, lambd, exp_dir, root_dir, wsmd)
                    data.append(res)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'methods.csv', index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(main)