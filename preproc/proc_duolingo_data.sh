#! /bin/bash

python duolingo_word_feats.py && python duolingo_seq_feats.py && python load_duolingo_word_feats.py && python load_duolingo_seq_feats.py && python make_vocab_dict.py
