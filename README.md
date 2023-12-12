# WSMD

> [Improving word mover’s distance by leveraging self-attention matrix](https://arxiv.org/abs/2211.06229)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/), [Sho Yokoi](https://www.cl.ecei.tohoku.ac.jp/~yokoi/), [Hidetoshi Shimodaira](http://stat.sys.i.kyoto-u.ac.jp/members/shimo/)                 
> *EMNLP 2023 Findings*

Word Mover's Distance (WMD) does not utilize word order, making it challenging to distinguish sentences with significant overlaps of similar words, even if they are semantically very different. Here, we attempt to improve WMD by incorporating the sentence structure represented by BERT's self-attention matrix (SAM). The proposed method is based on the Fused Gromov-Wasserstein distance, which simultaneously considers the similarity of the word embedding and the SAM for calculating the optimal transport between two sentences.

An illustration of OT for word embeddings
from sentence 1 to sentence 2.
![Fig. 1](assets/wsmd_explanation.png)

An illustration of OT for SAMs from sentence 1 to sentence 2.
![Fig. 2](assets/obama_sam.png)

Note that the camera-ready version, i.e. version 2 on arXiv, is a significant update to version 1 on arXiv.

# Code
The source code is being organized and will be available soon. 

# Citation
If you find our code or model useful in your research, please cite our paper:
```
@inproceedings{yamagiwa-etal-2023-improving,
    title = "Improving word mover{'}s distance by leveraging self-attention matrix",
    author = "Yamagiwa, Hiroaki  and
      Yokoi, Sho  and
      Shimodaira, Hidetoshi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.746",
    pages = "11160--11183",
    abstract = "Measuring the semantic similarity between two sentences is still an important task. The word mover{'}s distance (WMD) computes the similarity via the optimal alignment between the sets of word embeddings. However, WMD does not utilize word order, making it challenging to distinguish sentences with significant overlaps of similar words, even if they are semantically very different. Here, we attempt to improve WMD by incorporating the sentence structure represented by BERT{'}s self-attention matrix (SAM). The proposed method is based on the Fused Gromov-Wasserstein distance, which simultaneously considers the similarity of the word embedding and the SAM for calculating the optimal transport between two sentences. Experiments demonstrate the proposed method enhances WMD and its variants in paraphrase identification with near-equivalent performance in semantic textual similarity.",
}
```