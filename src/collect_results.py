from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def main(model_name='bert-base-uncased', emb_layer=0,
         attn_layer=6,
         attn_layers=[4, 5, 6, 7, 8, 9, 10, 11],
         task='PAWSWiki', dataset='sample', full=False,
         root_dir='/working'):

    root_dir = Path(root_dir)
    full_str = 'full_' if full else ''

    if model_name == 'distilbert-base-uncased':
        layer_num = 6
    else:
        layer_num = 12

    if task == 'STSB':
        metric = 'spearman'
    else:
        metric = 'auc'

    if task == 'STSB':
        # when you remove stopwords, some sentences become empty for STSB.
        # so, we need to remove these sentences from the evaluation.
        # here, the WMD-uniform-euclid method is used as a reference.
        method_dir = root_dir / f'output/eval_baseline/STSB/{dataset}/'\
            f'{model_name.replace("/", "_")}/'\
            f'emb_layer{emb_layer}/wmd_uniform_euclid'
        sims_path = method_dir / 'sims.pkl'
        with sims_path.open('rb') as f:
            sims = pickle.load(f)
        _inf_idx_1d = np.isneginf(sims)
        # duplicate to axis=1
        _inf_idx = np.repeat(_inf_idx_1d, 12*layer_num
                             ).reshape(-1, 12*layer_num)

    # baseline and smd
    merge_df = pd.DataFrame()
    input_dir = root_dir / f'output/{full_str}eval_baseline/{task}/'\
        f'{dataset}/{model_name.replace("/", "_")}/emb_layer{emb_layer}'
    for method_dir in input_dir.iterdir():

        if method_dir.is_file():
            continue

        sims_path = method_dir / 'sims.pkl'
        with sims_path.open('rb') as f:
            sims = pickle.load(f)

        gs_scores_path = method_dir / 'gs_scores.pkl'
        with gs_scores_path.open('rb') as f:
            gs_scores = pickle.load(f)

        input_csv = method_dir / 'result.csv'
        df = pd.read_csv(input_csv)

        if task == 'STSB':
            df = df.drop(columns=['pearson', 'spearman'])
            sims = sims[~_inf_idx_1d]
            gs_scores = gs_scores[~_inf_idx_1d]

        if metric == 'auc':
            score = roc_auc_score(gs_scores, sims)
        elif metric == 'pearson':
            score = pearsonr(gs_scores, sims)[0]
        elif metric == 'spearman':
            score = spearmanr(gs_scores, sims)[0]
        else:
            raise ValueError(f'Invalid metric: {metric}')
        df[metric] = score
        merge_df = pd.concat([merge_df, df])

    # smd and wsmd
    method_dirs = []
    for dir_name in ['smd', 'wsmd']:
        input_dir = root_dir / f'output/{full_str}eval_{dir_name}/{task}/'\
            f'{dataset}/{model_name.replace("/", "_")}/emb_layer{emb_layer}'
        for method_dir in input_dir.iterdir():
            # directory check
            if method_dir.is_file():
                continue
            method_dirs.append(method_dir)


    for method_dir in method_dirs:
        sims_all_path = method_dir / 'sims_all.pkl'
        if not sims_all_path.exists():
            continue
        with sims_all_path.open('rb') as f:
            sims_all = pickle.load(f)

        gs_scores_path = method_dir / 'gs_scores.pkl'
        with gs_scores_path.open('rb') as f:
            gs_scores = pickle.load(f)

        sims_all = np.array(sims_all)
        gs_scores = np.array(gs_scores)

        if task == 'STSB':
            sims_all = sims_all[~_inf_idx].reshape(-1, 144)
            gs_scores = gs_scores[~_inf_idx_1d]

        input_csv = method_dir / 'result.csv'

        # calculate wsmd for layer
        df = pd.read_csv(input_csv)
        df['attn_layer'] = str([attn_layer])
        sims = sims_all[:, 12*attn_layer:12*(attn_layer+1)]
        sims_mean = sims.mean(axis=1)

        if task == 'STSB':
            df = df.drop(columns=['pearson', 'spearman'])

        if metric == 'auc':
            score = roc_auc_score(gs_scores, sims_mean)
        elif metric == 'pearson':
            score = pearsonr(gs_scores, sims_mean)[0]
        elif metric == 'spearman':
            score = spearmanr(gs_scores, sims_mean)[0]
        else:
            raise ValueError(f'Invalid metric: {metric}')

        df[metric] = score
        merge_df = pd.concat([merge_df, df])

        # calculate wsmd for all layers from 4th layer
        df = pd.read_csv(input_csv)
        if task == 'STSB':
            df = df.drop(columns=['pearson', 'spearman'])

        df['attn_layer'] = str(attn_layers)
        sims = []
        for layer in attn_layers:
            assert layer < layer_num
            sims.append(sims_all[:, 12*layer:12*(layer+1)])
        sims = np.array(sims).mean(axis=0).mean(axis=1)
        if metric == 'auc':
            score = roc_auc_score(gs_scores, sims)
        elif metric == 'pearson':
            score = pearsonr(gs_scores, sims)[0]
        elif metric == 'spearman':
            score = spearmanr(gs_scores, sims)[0]

        df[metric] = score
        merge_df = pd.concat([merge_df, df])

        # calculate wsmd for all layers
        df = pd.read_csv(input_csv)
        if task == 'STSB':
            df = df.drop(columns=['pearson', 'spearman'])
        df['attn_layer'] = str(list(range(layer_num)))
        sims = sims_all.mean(axis=1)
        if metric == 'auc':
            score = roc_auc_score(gs_scores, sims)
        elif metric == 'pearson':
            score = pearsonr(gs_scores, sims)[0]
        elif metric == 'spearman':
            score = spearmanr(gs_scores, sims)[0]

        df[metric] = score
        merge_df = pd.concat([merge_df, df])

    merge_df = merge_df.reset_index(drop=True)
    # duplicaton row check
    if merge_df.duplicated().any():
        # print(merge_df[merge_df.duplicated()])
        raise ValueError('duplicated row exists')
    output_csv = root_dir / f'output/{full_str}eval-{task}-{dataset}-{model_name.replace("/", "_")}-emb_layer{emb_layer}.csv'  # noqa
    merge_df.to_csv(
        output_csv, index=False)

if __name__ == '__main__':
    import fire
    fire.Fire(main)