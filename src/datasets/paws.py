from collections import Counter

import numpy as np
import pandas as pd


def data_all_set(data_name, data_loader, data_type=None):
    refs = []
    cands = []
    gs_scores = []
    vocab_count = Counter()
    if data_name == 'PAWSWiki':
        assert data_type in ('dev', 'sample', 'test')
        evaluation = data_loader('/working/data/datasets/PAWSWiki')
    else:
        assert data_type in ('train', 'dev_and_test')
        evaluation = data_loader('/working/data/datasets/PAWSQQP')
    data_test = evaluation.data[data_type]
    rf, cd, gs = data_test
    if data_name == 'PAWSWiki' or data_name == 'PAWSQQP':
        rf = rf[:1500]
        cd = cd[:1500]
        gs = gs[:1500]

    for sent in rf+cd:
        vocab_count.update(sent)
    rf = [' '.join(x) for x in rf]
    cd = [' '.join(x) for x in cd]
    refs.extend(rf)
    cands.extend(cd)
    gs_scores.extend(gs)

    return refs, cands, gs_scores, vocab_count


def data_all_full_set(data_name, data_loader, data_type=None):
    refs = []
    cands = []
    gs_scores = []
    vocab_count = Counter()
    if data_name == 'PAWSWiki':
        assert data_type in ('dev', 'sample', 'test')
        evaluation = data_loader('/working/data/datasets/PAWSWiki')
    else:
        assert data_type in ('train', 'dev_and_test')
        evaluation = data_loader('/working/data/datasets/PAWSQQP')
    data_test = evaluation.data[data_type]
    rf, cd, gs = data_test

    for sent in rf+cd:
        vocab_count.update(sent)
    rf = [' '.join(x) for x in rf]
    cd = [' '.join(x) for x in cd]
    refs.extend(rf)
    cands.extend(cd)
    gs_scores.extend(gs)

    return refs, cands, gs_scores, vocab_count


class PAWSWiki(object):
    def __init__(self, task_path):
        self.datasets = ['train', 'dev', 'test', 'sample']
        self.loadFile(task_path)

    def loadFile(self, fpath):
        self.data = {}
        for dataset in self.datasets:
            df = pd.read_table(fpath + f'/{dataset}.tsv')
            sent1 = df.sentence1.values
            sent2 = df.sentence2.values
            gs_scores = df.label.values

            sent1 = [s.split() for s in sent1]
            sent2 = [s.split() for s in sent2]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))
            self.data[dataset] = (sent1, sent2, gs_scores)


class PAWSQQP(object):
    def __init__(self, task_path):
        self.datasets = ['train', 'dev_and_test']
        self.loadFile(task_path)

    def loadFile(self, fpath):
        self.data = {}
        for dataset in self.datasets:
            df = pd.read_table(fpath + f'/{dataset}.tsv')
            sent1 = df.sentence1.values
            sent2 = df.sentence2.values
            gs_scores = df.label.values

            sent1 = [s.split() for s in sent1]
            sent2 = [s.split() for s in sent2]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))
            self.data[dataset] = (sent1, sent2, gs_scores)


def main(full=True):
    task = 'PAWSQQP'
    task_func = PAWSQQP
    dataset = 'train'
    if full:
        _, _, gs_scores, _ = data_all_full_set(task, task_func, dataset)
    else:
        raise NotImplementedError
    print(sum(gs_scores), len(gs_scores),
          f'{sum(gs_scores)/len(gs_scores)*100:.2f}%')


if __name__ == '__main__':
    main()
