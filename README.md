# BiLSTM-CRF-for-sequence-tagging
Implementation of [BiLSTM-CRF for sequence tagging](https://arxiv.org/pdf/1508.01991.pdf)

Requirements:
======
    Pytorch
    numpy
    pickle
    sklearn

Usage:
======
## NER:
This command will run the model on a gpu using CUDA, if you would like to run the model without CUDA then in ner_train.py change the 'use_gpu = 1' line after the imports at the top to 'use_gpu = 0'

  python ner_train.py

## POS:
Blah blah


To prepare sequential duolingo dataset
    cd preproc/
    python duolingo_split.py


To train from duolingo dataset,
    python duolingo_train.py

For post-hoc analysis on duolingo
    python duolingo_analysis.py
