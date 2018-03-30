# BiLSTM-CRF-for-sequence-tagging
Implementation of [BiLSTM-CRF for sequence tagging](https://arxiv.org/pdf/1508.01991.pdf)


To prepare sequential duolingo dataset
    cd preproc/
    python duolingo_split.py
    
    
To train from duolingo dataset, 
    python duolingo_train.py
    
For post-hoc analysis on duolingo
    python duolingo_analysis.py