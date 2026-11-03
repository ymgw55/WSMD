# Third-Party Notices

- `src/modules/rots.py`, `src/modules/owmd.py`, `src/modules/wmdo.py`: not included in this repository. `setup/setup_third_party_modules.sh` builds them from [rots](https://github.com/zihao-wang/rots) (commit `27f8214`), [SentenceMatching](https://github.com/BangLiu/SentenceMatching) (commit `4b8c194`, GPL-3.0), and [WMDO](https://github.com/julianchow/WMDO) (commit `f1716cc`), and applies our changes in `patches/third_party_modules.patch`.
- `src/datasets/sts.py`: data loading adapted from [SentEval](https://github.com/facebookresearch/SentEval). [BSD License](third_party_licenses/SentEval-LICENSE).
- `src/save_model.py`: `truncate()` and `padding()` adapted from [BERTScore](https://github.com/Tiiiger/bert_score). [MIT License](third_party_licenses/bert_score-LICENSE).
