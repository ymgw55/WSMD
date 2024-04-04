import pickle
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.ot_utils import rm_stopwords
from utils import get_logger

warnings.filterwarnings("ignore")


def main(model_name='bert-base-uncased', root_dir='/working'):

    root_dir = Path(root_dir)

    assert model_name in ('bert-base-uncased',
                          'roberta-base',
                          'princeton-nlp/unsup-simcse-bert-base-uncased',
                          'distilbert-base-uncased',
                          'google/multiberts-seed_0')

    output_dir = root_dir / f'output/VBA/{model_name.replace("/", "_")}'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{model_name.replace("/", "_")}.pkl'

    log_file = str(output_dir / f'{model_name.replace("/", "_")}.log')
    logger = get_logger(log_file=log_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    space_token = 'Ġ'

    stop_words = stopwords.words('english')
    punct = [i for i in string.punctuation]
    stop_words = stop_words + punct + \
        [cls_token, sep_token, pad_token, space_token]
    stop_words = set(stop_words)

    result_dict = {}
    task = 'PAWSQQP'
    datasets = ['train', 'dev_and_test']

    for dataset in datasets:
        logger.info(f'task: {task}')
        logger.info(f'dataset: {dataset}')

        model_dir = root_dir / f'data/whiten_model/{task}/'\
            f'{dataset}/{model_name.replace("/", "_")}'
        with open(model_dir / 'results.pkl', 'rb') as f:
            results = pickle.load(f)
        sent1s_results, sent2s_results = results

        attns_shapes = sent1s_results[1][0].shape
        layers = attns_shapes[0]
        heads = attns_shapes[1]
        attns = [[[] for _ in range(heads)] for _ in range(layers)]

        # layer and head dicision
        for (_, sent1_attns, sent1_tokens,
                _, sent2_attns, sent2_tokens) in tqdm(list(
                zip(*sent1s_results, *sent2s_results))):
            for layer in range(layers):
                for head in range(heads):

                    sent1_sam = sent1_attns[layer][head]
                    sent2_sam = sent2_attns[layer][head]                        
                    attns[layer][head].append(sent1_sam)
                    attns[layer][head].append(sent2_sam)

        attns_mean = [[None for _ in range(heads)] for _ in range(layers)]
        for layer in tqdm(range(layers)):
            for head in range(heads):
                n = max([len(sam) for sam in attns[layer][head]])
                attns_mean[layer][head] = np.zeros((n, n))
                L = len(attns[layer][head])
                for sam in attns[layer][head]:
                    attns_mean[layer][head][
                        :len(sam), :len(sam)] += sam / L

        # calculate variablity of sam
        attns_var = [[None for _ in range(heads)] for _ in range(layers)]
        for layer in tqdm(range(layers)):
            for head in range(heads):
                denom = 0
                numerator = 0
                for sam in attns[layer][head]:
                    denom += np.sum(sam)
                    numerator += np.sum(
                        np.abs(sam - attns_mean[layer][head][
                            :len(sam), :len(sam)]))
                denom *= 2
                attns_var[layer][head] = numerator / denom

        # find the most variable layer
        data = []
        for layer in range(layers):
            for head in range(heads):
                data.append({
                    'layer': layer,
                    'head': head,
                    'stat': attns_var[layer][head]
                })

        max_attn_layer = np.argmax(
            [np.mean(attns_var[layer]) for layer in range(layers)])

        df = pd.DataFrame(data)
        df.to_csv(output_dir / f'{task}_{dataset}.csv', index=False)

        logger.info(f'max_attn_layer: {max_attn_layer}')
        result_dict[(task, dataset)] = max_attn_layer

    logger.info(result_dict)
    with open(output_path, 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    import fire
    fire.Fire(main)