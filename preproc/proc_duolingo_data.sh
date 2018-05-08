#! /bin/bash

mkdir -p ../data/duolingo/vocabs_es_en
mkdir -p ../data/duolingo/vocabs_en_es
mkdir -p ../data/duolingo/vocabs_fr_en
mkdir -p ../models/duolingo_models
mkdir -p ../duolingo_text
mkdir -p ../embeddings/

python duolingo_word_feats.py --lang es_en && python duolingo_seq_feats.py --lang es_en && python load_duolingo_word_feats.py --lang es_en && python load_duolingo_seq_feats.py --lang es_en && python make_vocab_dict.py --lang es_en && python duolingo_word_feats.py --lang en_es && python duolingo_seq_feats.py --lang en_es && python load_duolingo_word_feats.py --lang en_es && python load_duolingo_seq_feats.py --lang en_es && python make_vocab_dict.py --lang en_es && python duolingo_word_feats.py --lang fr_en && python duolingo_seq_feats.py --lang fr_en && python load_duolingo_word_feats.py --lang fr_en && python load_duolingo_seq_feats.py --lang fr_en && python make_vocab_dict.py --lang fr_en
