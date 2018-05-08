Second Language Acquisition Modeling on Duolingo Data
======
BiLSTM (+/-) CNN (+/-) CRF
Network based on [BiLSTM-CRF for sequence tagging](https://arxiv.org/pdf/1508.01991.pdf)


Requirements:
======
    Python 3.6

    Pytorch
    numpy
    pickle
    sklearn


Usage:
======
1. Download the duolingo data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO and extract into data/data_duolingo/ (create directory if it doesn't exist)

2. change to the preproc/ directory and run ./proc_duolingo_data.sh

3. Run (from base project directory):
python duolingo_train.py --lang fr_en --gpu 0 --lr 0.05            # for cpu
OR
python duolingo_train.py --lang fr_en --gpu 1 --device 2 --lr 0.05 # for gpu, --device can be omitted

Change the flags to run with different language pairs, gpu options, or learning rate


Code
======
duolingo_train.py contains code which runs a BLSTM (+/-) CNN (+/-) CRF on the Duolingo dataset.
models/duolingo_model.py contains the BLSTM (+/-) CNN
models/crf.py contains the optional CRF
preproc/ contains files to preprocess the data
