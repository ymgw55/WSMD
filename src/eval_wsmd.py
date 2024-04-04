import pickle
import string
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
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


def sim_calculate(
        method,
        sent1_emb, sent1_sam, sent1_tokens, sent1_t2w, sent1_parsing_data,
        sent2_emb, sent2_sam, sent2_tokens, sent2_t2w, sent2_parsing_data,
        wsmd, word2weight, cost_name, params, stop_words, tokenizer, lambd):

    # BUG: some emb not writable
    sent1_emb = sent1_emb.copy()
    sent2_emb = sent2_emb.copy()

    if cost_name == 'swd':
        # normaize emb
        sent1_emb = sent1_emb / np.linalg.norm(
            sent1_emb, axis=1, keepdims=True)
        sent2_emb = sent2_emb / np.linalg.norm(
            sent2_emb, axis=1, keepdims=True)

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

    def get_t2w_parsing_data(sent_tokens, tokenizer):
        sent_t2w, sent_word_seq = tokens2words(sent_tokens, tokenizer)
        sent_parsing_data = nlp('\n\n'.join([' '.join(sent_word_seq)]))
        sent_parsing_data = stanza2dic_sent(sent_parsing_data)
        assert len(sent_parsing_data) == len(sent_t2w)
        return sent_t2w, sent_parsing_data

    sims = []
    sims_all = []
    for (sent1_embs, sent1_attns, sent1_tokens,
            sent2_embs, sent2_attns, sent2_tokens) in tqdm(list(
            zip(*sent1s_results, *sent2s_results))):

        sent1_emb = sent1_embs[emb_layer]
        sent2_emb = sent2_embs[emb_layer]

        layers = len(sent1_attns)
        heads = len(sent1_attns[0])

        # #  single thread
        # tmp = []
        # for attn_layer in range(layers):
        #     for attn_head in range(heads):
        #         sent1_sam = sent1_attns[attn_layer][attn_head]
        #         sent2_sam = sent2_attns[attn_layer][attn_head]

        #         sim = sim_calculate(
        #             method,
        #             sent1_emb, sent1_sam, sent1_tokens,
        #             sent2_emb, sent2_sam, sent2_tokens,
        #             wsmd=wsmd, word2weight=word2weight, cost_name=cost_name,
        #             params=params, stop_words=stop_words, tokenizer=tokenizer,
        #             lambd=lambd)
        #         tmp.append(sim)

        # multi thread
        tmp = joblib.Parallel(n_jobs=24)(
            joblib.delayed(sim_calculate)(
                method,
                sent1_emb, sent1_attns[attn_layer][attn_head], sent1_tokens,
                *get_t2w_parsing_data(sent1_tokens, tokenizer),
                sent2_emb, sent2_attns[attn_layer][attn_head], sent2_tokens,
                *get_t2w_parsing_data(sent2_tokens, tokenizer),
                wsmd=wsmd, word2weight=word2weight, cost_name=cost_name,
                params=params, stop_words=stop_words, tokenizer=tokenizer,
                lambd=lambd)
            for attn_layer in range(layers) for attn_head in range(heads))

        sims_all.append(tmp)
        sims.append(np.mean(tmp))

    # replace -inf with min value
    sims = np.array(sims)
    min_v = np.min(sims[np.isfinite(sims)])
    sims[np.isneginf(sims)] = min_v

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

    # save tmp and gs_scores
    with open(exp_dir / 'sims_all.pkl', 'wb') as f:
        pickle.dump(sims_all, f)
    with open(exp_dir / 'gs_scores.pkl', 'wb') as f:
        pickle.dump(gs_scores, f)

    return res_dict


def main(model_name, emb_layer,
         task='PAWSWiki', dataset='test',
         whiten=True, normalize=False, full=False, lambd=0.5,
         root_dir='/working'):

    root_dir = Path(root_dir)

    assert model_name in ('bert-base-uncased',
                          'roberta-base',
                          'princeton-nlp/unsup-simcse-bert-base-uncased',
                          'distilbert-base-uncased',
                          'google/multiberts-seed_0')

    if full:
        output_dir = root_dir / f'output/full_eval_wsmd/{task}/{dataset}'
    else:
        output_dir = root_dir / f'output/eval_wsmd/{task}/{dataset}'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    output_dir = output_dir / model_name.replace("/", "_") \
        / f'emb_layer{emb_layer}'

    methods = ['wmd', 'synwmd', 'wrd']
    for method in methods:
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
                for wsmd in [True]:
                    if wsmd and method in ('opwd', 'smd'):
                        continue
                    if wsmd:
                        exp = f'{method}_wsmd_{weight_name}'
                    else:
                        exp = f'{method}_{weight_name}'
                    if cost_name is not None:
                        exp += f'_{cost_name}'

                    exp_dir = output_dir / exp

                    if (exp_dir / 'result.csv').exists():
                        df = pd.read_csv(exp_dir / 'result.csv')
                        res = df.iloc[0].to_dict()
                        print(res)
                        data.append(res)
                        continue
                    print(exp)
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    res = ot_method(
                        method, rm_sw, weight_name, cost_name, params,
                        task, dataset, model_name, emb_layer,
                        whiten, normalize, full, lambd, exp_dir, root_dir,
                        wsmd)
                    print(res)
                    data.append(res)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'methods.csv', index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(main)