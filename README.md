# aist-repo

The source code for AIST 2021 submission.

The provided code is exclusively for the reviewers only and should not be shared with anyone else. We will separately share the resources publically under the appropriate license after the acceptance of the paper.


### Repository structure

```
├── src/ -- Source code
│    ├── AbaeLexRank.py -- LexRank variation for ABAE model 
│    ├── ArtmLexRank.py -- LexRank variation for BigARTM model
│    ├── data_reader.py -- Json reader
│    ├── metrics.py -- Metrics computing (ROUGE, precision)
│    ├── main.py -- Functions for the running
│    └── rus_preprocessing_udpipe.py -- Preprocessing with UDPipe
└── wiki_persons_data.zip -- The source data from Russian Wikipedia
```
ABAE implementation was taken from [Attention-Based-Aspect-Extraction](https://github.com/madrugado/Attention-Based-Aspect-Extraction)

UDPipe preprocessing code was taken from [webvectors](https://github.com/akutuzov/webvectors/blob/master/preprocessing/rus_preprocessing_udpipe.py)
