import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import get_logger


def truncate(a, tokenizer):
    tokens = tokenizer.tokenize(a)
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0:(tokenizer.model_max_length - 2)]
    return tokens


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def model_encode(model, x, attention_mask, device):
    with torch.no_grad():
        result = model(x.to(device), attention_mask=attention_mask.to(device),
                       output_attentions=True, output_hidden_states=True)
    embeddings = result['hidden_states']
    attentions = result['attentions']
    return embeddings, attentions


def get_inputs(sents, tokenizer):

    pad = tokenizer.pad_token
    sep = tokenizer.sep_token
    cls = tokenizer.cls_token
    tokens_to_ids = tokenizer.convert_tokens_to_ids

    tokens = [[cls]+truncate(sent, tokenizer)+[sep] for sent in sents]
    ids = [tokens_to_ids(token) for token in tokens]
    pad_token = tokens_to_ids([pad])[0]
    padded, lens, mask = padding(ids, pad_token, dtype=torch.long)

    return padded, lens, mask, tokens


def get_outputs(sents, model, tokenizer, device):

    padded_sens, lens, mask, tokens = get_inputs(sents, tokenizer)

    with torch.no_grad():
        embedding, attention = \
            model_encode(model, padded_sens, mask, device)
    embedding = torch.stack(embedding)
    attention = torch.stack(attention)
    embedding = embedding.swapaxes(0, 1)
    attention = attention.swapaxes(0, 1)

    embedding = embedding.cpu().numpy()
    embedding = [embedding[i] for i in range(len(embedding))]
    attention = attention.cpu().numpy()
    attention = [attention[i] for i in range(len(attention))]
    mask = mask.cpu().numpy()
    mask = [mask[i] for i in range(len(mask))]
    lens = lens.cpu().numpy()
    lens = [lens[i] for i in range(len(lens))]

    return embedding, attention, mask, lens, tokens


def get_sents_results(sents, model, tokenizer, batch_size, device):
    sents_embs = []
    sents_attns = []
    sents_masks = []
    sents_lens = []
    sents_tokens = []
    for b_start in tqdm(range(0, len(sents), batch_size)):
        b_sents = sents[b_start:b_start+batch_size]

        b_sents_embs, b_sents_attns, b_sents_masks, \
            b_sents_lens, b_sents_tokens,  = \
            get_outputs(b_sents, model, tokenizer, device)
        sents_embs += b_sents_embs
        sents_attns += b_sents_attns
        sents_masks += b_sents_masks
        sents_lens += b_sents_lens
        sents_tokens += b_sents_tokens

    sents_no_pad_embs = []
    sents_no_pad_attns = []
    for sent_emb, sent_attn, sent_len in zip(sents_embs, sents_attns,
                                             sents_lens):
        sent_emb = sent_emb[:, :sent_len, :]
        sent_attn = sent_attn[:, :, :sent_len, :sent_len]
        sents_no_pad_embs.append(sent_emb)
        sents_no_pad_attns.append(sent_attn)

    return sents_no_pad_embs, sents_no_pad_attns, sents_tokens


def get_results(sent1s, sent2s, model, tokenizer, batch_size, device):
    sent1s_results = get_sents_results(sent1s, model, tokenizer, batch_size,
                                       device)
    sent2s_results = get_sents_results(sent2s, model, tokenizer, batch_size,
                                       device)
    return sent1s_results, sent2s_results


def compute_params(vecs):
    vecs = np.array(vecs)
    mu = vecs.mean(axis=0, keepdims=True)
    params = {'mu': mu}
    cov = np.cov((vecs - mu).T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    params['W'] = W
    return params


def transform(vecs, params):
    mu = params['mu']
    W = params['W']
    if W is None:
        vecs = vecs - mu
    else:
        vecs = (vecs - mu).dot(W)
    return vecs


def main(full=False, root_dir='/working'):

    root_dir = Path(root_dir)

    model_names = (
        'bert-base-uncased',
        'roberta-base',
        'princeton-nlp/unsup-simcse-bert-base-uncased',
        'distilbert-base-uncased',
        'google/multiberts-seed_0',
    )

    logger = get_logger()

    for model_name in model_names:
        logger.info(f'model_name: {model_name}')

        tasks = ['PAWSQQP', 'PAWSWiki', 'STSB', 'SICK']
        for task in tasks:
            logger.info(f'task: {task}')

            if task == 'STSB':
                from datasets.sts import STSB, data_all_set
                task_func = STSB
                batch_size = 256
                datasets = ['sts-test', 'sts-sample']
            elif task == 'PAWSWiki':
                if full:
                    from datasets.paws import PAWSWiki
                    from datasets.paws import data_all_full_set as data_all_set
                else:
                    from datasets.paws import PAWSWiki, data_all_set
                task_func = PAWSWiki
                batch_size = 256
                datasets = ['dev', 'test', 'sample']
            elif task == 'PAWSQQP':
                if full:
                    from datasets.paws import PAWSQQP
                    from datasets.paws import data_all_full_set as data_all_set
                else:
                    from datasets.paws import PAWSQQP, data_all_set
                task_func = PAWSQQP
                batch_size = 64
                datasets = ['train', 'dev_and_test']
            elif task == 'SICK':
                from datasets.sick import SICK, data_all_set
                task_func = SICK
                batch_size = 256
                datasets = ['SICK_trial', 'SICK_test_annotated']

            logger.info(f'batch_size: {batch_size}')
            logger.info(f'datasets: {datasets}')

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()

            for datasets in datasets:
                sent1s, sent2s, _, _ = data_all_set(task, task_func, datasets)
                if full:
                    output_dir = root_dir /\
                        f'data/full_whiten_model/{task}/'\
                        f'{datasets}/{model_name.replace("/", "_")}'
                else:
                    output_dir = root_dir /\
                        f'data/whiten_model/{task}/'\
                        f'{datasets}/{model_name.replace("/", "_")}'

                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'output_dir: {output_dir}')

                sent1s_results, sent2s_results = get_results(
                    sent1s, sent2s, model, tokenizer, batch_size, device)

                layers = sent1s_results[0][0].shape[0]
                layer_embs = [[] for _ in range(layers)]

                for (sent1_embs, _, _, sent2_embs, _, _) in tqdm(list(
                        zip(*sent1s_results, *sent2s_results))):

                    for layer in range(layers):
                        for i in range(len(sent1_embs[layer])):
                            layer_embs[layer].append(sent1_embs[layer][i])
                        for i in range(len(sent2_embs[layer])):
                            layer_embs[layer].append(sent2_embs[layer][i])

                params_list = []
                for layer in range(layers):
                    params = compute_params(
                        layer_embs[layer])
                    params_list.append(params)

                sent1s_embs = []
                sent2s_embs = []
                for (sent1_embs, _, _, sent2_embs, _, _) in tqdm(list(
                        zip(*sent1s_results, *sent2s_results))):

                    tmp_sent1_embs = []
                    tmp_sent2_embs = []
                    for layer in range(layers):
                        tmp_sent1_embs.append(
                            transform(sent1_embs[layer], params_list[layer]))
                        tmp_sent2_embs.append(
                            transform(sent2_embs[layer], params_list[layer]))

                    sent1s_embs.append(np.array(tmp_sent1_embs))
                    sent2s_embs.append(np.array(tmp_sent2_embs))

                sent1s_results = \
                    sent1s_embs, sent1s_results[1], sent1s_results[2]
                sent2s_results = \
                    sent2s_embs, sent2s_results[1], sent2s_results[2]

                results = sent1s_results, sent2s_results
                with open(output_dir / 'results.pkl', 'wb') as f:
                    pickle.dump(results, f)


if __name__ == '__main__':
    main()
    main(full=True)