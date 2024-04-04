from collections import defaultdict

import networkx as nx
import numpy as np

from modules.ot_distances import get_WRD_cost
from modules.ot_utils import stanza2dic, stanza2dic_sent


def sentdic2undicgraph(sentDic):
    # convert to undirectional graph
    G = nx.Graph()
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((d_word_idx, h_word_idx))
    G.add_edges_from(edge_list)
    G.remove_node(-1)  # remove root
    return G


def build_graph(parsing_data, word2id, hop_size=3):
    num_pair = 0
    G = nx.Graph()
    edge_count = defaultdict(float)
    total_count = {id: 0 for id in range(len(word2id))}
    for sent in parsing_data:
        tree = sentdic2undicgraph(sent)
        for word_idx, word in enumerate(sent):
            d_word_idx = word2id[word['text']]
            neighb_dict = nx.single_source_shortest_path_length(
                tree, word_idx, cutoff=hop_size)
            # adding co-occurrence time
            for neighb_idx, hop in neighb_dict.items():
                if hop == 0:
                    # avoid the word itself
                    continue
                h_word_idx = word2id[sent[neighb_idx]['text']]
                edge_count[(d_word_idx, h_word_idx)] += 1 / hop
                total_count[d_word_idx] += 1 / hop
                num_pair += 1
    # normalize
    # edge_count = {x:c/total_count[x[0]] for x, c in edge_count.items()}
    weight_edge_list = [x+tuple([c]) for x, c in edge_count.items()]
    G.add_weighted_edges_from(weight_edge_list)
    # print('num_pair:', num_pair)
    return G, num_pair


def get_swf_weight(word2id, words, vocab, nlp):
    # Parser
    words = [w for w in words if not (w.isspace() or len(w) == 0)]
    parsing_data = nlp('\n\n'.join(words))
    parsing_data = stanza2dic(parsing_data)

    G, _ = build_graph(parsing_data, word2id, hop_size=3)
    word2weight = {}
    pr = nx.pagerank(G, alpha=0.2)
    for k in range(len(vocab)):
        if k in pr:
            word2weight[vocab[k]] = 1/(pr[k])
    return word2weight


def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
            if length == n]


def neighborhood_v2(G, node, n):
    neighb_dict = nx.single_source_shortest_path_length(G, node, cutoff=n)

    neighbors = []
    for neighb_idx, hop in neighb_dict.items():
        if hop == 0:
            continue  # avoid the word itself
        neighbors.append(neighb_idx)
    return neighbors


def node_bottomsubtree(G):
    descendants = []
    for node in list(G.nodes()):
        dcdt = list(nx.descendants(G, node))
        if dcdt:
            dcdt.append(node)  # append the parent itself
            descendants.append(dcdt)
    return descendants


def node_smallsubtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        for hop in range(1, hop_num+1):
            subt = neighborhood_v2(G, node, hop)
            if subt:  # avoid only one node
                subt.append(node)  # append parent itself
                if subtree:
                    if subtree[-1] == subt:
                        break  # stop when no growing
                subtree.append(subt)
    return subtree


def node_exactsubtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        subt = neighborhood_v2(G, node, hop_num)
        if subt:  # avoid only one node
            subt.append(node)  # append parent itself
            if subtree:
                if subtree[-1] == subt:
                    break  # stop when no growing
            subtree.append(subt)
    return subtree


def node_centersubtree(G, hop_num=1):
    subtree = []
    for node in list(G.nodes()):
        subt = neighborhood_v2(G, node, hop_num)
        if subt:  # avoid only one node
            subt.append(node)  # append parent itself
            subtree.append(subt)
    return subtree


def node_topsubtree(G, source):
    subtree = []
    for hop in range(1, len(list(G.nodes()))):
        subt = neighborhood_v2(G, source, hop)
        subt.append(source)  # append parent itself
        if subtree:
            if subtree[-1] == subt:
                break  # stop when no growing
        subtree.append(subt)
    return subtree


def sentdic2dicgraph(sentDic, delete_list=[]):
    # convert to undirectional graph
    G = nx.DiGraph()
    edge_list = []
    for d_word_idx, word in enumerate(sentDic):
        h_word_idx = word["head"]-1
        edge_list.append((h_word_idx, d_word_idx))
    G.add_edges_from(edge_list)
    G.remove_node(-1)  # remove root
    return G


def form_subtree(parsed_sent, t2w, tree='s', rel_kept=[], hop_num=3):
    G = sentdic2dicgraph(parsed_sent)
    if tree == 'b':
        subtree_w = node_bottomsubtree(G)
    elif tree == 't':
        source = [id for id, element in enumerate(parsed_sent)
                  if element['deprel'] == 'root'][0]
        subtree_w = node_topsubtree(G, source)
    elif tree == 's':
        subtree_w = node_smallsubtree(G, hop_num)
    elif tree == 'e':
        subtree_w = node_exactsubtree(G, hop_num)
    elif tree == 'c':
        G = sentdic2undicgraph(parsed_sent)
        subtree_w = node_centersubtree(G, hop_num)
    else:
        subtree_w = []

    # convert to token id
    subtree_t = []
    for sbt in subtree_w:
        if len(rel_kept) > 0:
            if not parsed_sent[sbt[-1]]['deprel'] in rel_kept:
                continue
        subtree_t.append([idy for idx in sbt for idy in t2w[idx]])
        # subtree_t.append([t2w[idx][0] for idx in sbt])
    return subtree_t


def get_tree_emb(emb, subtree, w):
    te = []
    for tree_idx in subtree:
        embs = emb[tree_idx, :]
        ws = w[tree_idx]
        ws = ws / (np.sum(ws) + 1e-30)
        embs = embs * ws[:, None]
        sum_emb = embs.sum(axis=0)
        sum_emb = sum_emb / (np.linalg.norm(sum_emb) + 1e-30)
        te.append(sum_emb)
    return np.array(te)


def get_SWD_cost(sent1_emb, sent1_parsing_data, sent1_t2w, w1,
                 sent2_emb, sent2_parsing_data, sent2_t2w, w2,
                 a=1.0):
    dist_matrix = get_WRD_cost(sent1_emb, sent2_emb)
    sent1_subtree = form_subtree(sent1_parsing_data, sent1_t2w)
    sent2_subtree = form_subtree(sent2_parsing_data, sent2_t2w)
    tree1_emb = get_tree_emb(sent1_emb, sent1_subtree, w1)
    tree2_emb = get_tree_emb(sent2_emb, sent2_subtree, w2)
    dist_tree = get_WRD_cost(tree1_emb, tree2_emb)
    dist_count = {(id1, id2): [] for id1 in range(len(sent1_emb))
                  for id2 in range(len(sent2_emb))}
    for tree_id1, id1 in enumerate(sent1_subtree):
        for tree_id2, id2 in enumerate(sent2_subtree):
            tree_distance = dist_tree[tree_id1, tree_id2]
            for i1 in id1:
                for i2 in id2:
                    dist_count[(i1, i2)].append(tree_distance)
    for key, value in dist_count.items():
        idx = list(key)
        if len(value):
            dist = np.average(value)
        else:
            dist = 0
        dist_matrix[idx[0], idx[1]] += a * dist
    return dist_matrix


def test():
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from modules.ot_utils import nlp, stanza2dic, tokens2words

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # sample long sentence
    sents = ['The quick brown fox jumps over the lazy dog.',
              'The city sits at the confluence of the Snake River with the great Weiser River , which marks the border with Oregon .']  # noqa

    embs = []
    sent_parsing_datas = []
    sent_t2ws = []
    ws = []
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        with torch.no_grad():
            emb = model(torch.tensor([tokenizer.encode(sent)])
                        ).last_hidden_state[0]
        emb = emb.detach().cpu().numpy()
        embs.append(emb)

        sent_t2w, sent_word_seq = tokens2words(tokens, tokenizer)
        sent_parsing_data = nlp('\n\n'.join([' '.join(sent_word_seq)]))
        sent_parsing_data = stanza2dic_sent(sent_parsing_data)
        sent_parsing_datas.append(sent_parsing_data)
        sent_t2ws.append(sent_t2w)
        ws.append(np.ones(len(emb))/len(emb))
    sent1_emb, sent1_parsing_data, sent1_t2w, w1 = \
        embs[0], sent_parsing_datas[0], sent_t2ws[0], ws[0]
    sent2_emb, sent2_parsing_data, sent2_t2w, w2 = \
        embs[1], sent_parsing_datas[1], sent_t2ws[1], ws[1]

    C = get_SWD_cost(sent1_emb, sent1_parsing_data, sent1_t2w, w1,
                     sent2_emb, sent2_parsing_data, sent2_t2w, w2)

    print(C)


if __name__ == '__main__':
    test()
