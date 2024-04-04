import io
from collections import Counter

import numpy as np


def data_all_set(data_name, data_loader, data_type=None):
    refs = []
    cands = []
    gs_scores = []
    vocab_count = Counter()

    evaluation = data_loader('/working/data/datasets/SICK')
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


class SICK(object):

    def __init__(self, taskpath):
        self.datasets = ['SICK_train', 'SICK_trial', 'SICK_test_annotated']
        self.loadFile(taskpath)

    def loadFile(self, fpath):
        self.data = {}

        for dataset in self.datasets:

            # list for the first sentence
            sent1 = []
            # list for the second sentence
            sent2 = []
            gs_scores = []

            with io.open(fpath + '/%s.txt' % dataset, 'r',
                         encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    larr = line.split('\t')
                    sent1.append(larr[1])
                    sent2.append(larr[2])
                    gs_scores.append(float(larr[3]))

            sent1 = [s.split() for s in sent1]
            sent2 = [s.split() for s in sent2]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)


def test():
    data_name = 'SICK'
    data_loader = SICK
    data_type = 'SICK_test_annotated'
    refs, cands, gs_scores, _ = data_all_set(
        data_name, data_loader, data_type)
    print(sum(gs_scores), len(gs_scores),
          f'{sum(gs_scores)/len(gs_scores)*100:.2f}%')


if __name__ == '__main__':
    test()
