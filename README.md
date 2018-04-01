BiLSTM-CRF-for-sequence-tagging
======
Implementation of [BiLSTM-CRF for sequence tagging](https://arxiv.org/pdf/1508.01991.pdf)

Requirements:
======
    Pytorch
    numpy
    pickle
    sklearn

Usage:
======
This command will run the model on a gpu using CUDA, if you would like to run the model without CUDA then in ner_train.py or pos_train.py change the 'use_gpu = 1' line after the imports at the top to 'use_gpu = 0'.

## NER:
  python ner_train.py

## POS:
  python pos_train.py

## Data
  The data should be stored in the data directory in the following format:

  data
  ├── conll2000
  │   ├── test.txt
  │   └── train.txt
  ├── ner
  │   ├── eng.testa
  │   ├── eng.testb
  │   └── eng.train
  └── pos
      ├── dev.txt
      ├── test.txt
      └── train.txt

Code
======
ner_train.py and pos_train.py contain code which runs the same model on the data detailed above. The neural network code is in the models/ directory, where the relevant files are lstm_cnn.py, which containes the BLSTM-CNN-CRF main code and imports the CRF code from crf.py. The rest of the files are not used in running the final model.

Duolingo
======
To prepare sequential duolingo dataset
    cd preproc/
    python duolingo_split.py

To train from duolingo dataset,
    python duolingo_train.py

For post-hoc analysis on duolingo
    python duolingo_analysis.py
