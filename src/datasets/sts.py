import io
from collections import Counter

import numpy as np


def data_all_set(data_name, data_loader, data_type=None):
    refs = []
    cands = []
    gs_scores = []
    vocab_count = Counter()

    evaluation = data_loader('/working/data/datasets/STSB')
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


class STSB(object):

    def __init__(self, taskpath):
        self.datasets = ['sts-train', 'sts-dev', 'sts-test', 'sts-sample']
        self.loadFile(taskpath)

    def loadFile(self, fpath):
        self.data = {}

        for dataset in self.datasets:

            # list for the first sentence
            sent1 = []
            # list for the second sentence
            sent2 = []
            gs_scores = []

            with io.open(fpath + '/%s.csv' % dataset, 'r',
                         encoding='utf-8') as f:
                for i, line in enumerate(f):
                    larr = line.split('\t')
                    sent1.append(larr[5])
                    sent2.append(larr[6])
                    gs_scores.append(float(larr[4]))

            sent1 = [s.split() for s in sent1]
            sent2 = [s.split() for s in sent2]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)


