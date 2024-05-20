# FakeFlow in Spanish
This repository contains the code for the paper "Emotions and News Structure: an Analysis of the Language of Fake News in Spanish".

## FakeFlow: Fake News Detection by Modeling the Flow of Affective Information

### Data and resources

In order to reproduce our setting, make sure you first obtain the following data and resources, and store them in the correct location.

#### Dataset:
* [FakeNewsCorpusSpanish](https://github.com/jpposadas/FakeNewsCorpusSpanish): Open the files, and store them as `.tsv` files, tab-separated, in the `./data/fakedes/` folder.
    * [Training set](https://github.com/jpposadas/FakeNewsCorpusSpanish/blob/master/train.xlsx): save as `train.tsv`.
    * [Development set](https://github.com/jpposadas/FakeNewsCorpusSpanish/blob/master/development.xlsx): save as `development.tsv`.
    * [Test set](https://github.com/jpposadas/FakeNewsCorpusSpanish/blob/master/test.xlsx): save as `test.tsv`.

#### Linguistic resources:
* [Word2Vec embeddings from the SBWCE](https://crscardellino.ar/SBWCE/): download the word vectors in text format ([download link](https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2)) and unzip the file. You'll be able to specify the path to the file when calling the model.
* [NRC Emotions Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm): download the NRC Word-Emotion Association Lexicon, unzip it. Go to NRC-Emotion-Lexicon/OneFilePerLanguage and then search for `Spanish-NRC-EmoLex.txt` file and save it in `./features/emotional/Spanish-NRC-EmoLex.txt`.
* [Imageability from Guasch et al. (2015)](https://link.springer.com/article/10.3758/s13428-015-0684-y#Sec13): download the ESM1 dataset from the 'Electronic supplementary material' section, and save it as a csv as `./features/imageability/imageability_es.csv`.
* [HurtLex Lexicon](https://github.com/valeriobasile/hurtlex/tree/master/lexica/ES/1.2): download the file and store it as `./features/hurtful/hurtlex_ES.tsv`.

#### Expected directory structure:

To run the code for our experiments, we assume the following directory structure:

```bash
FakeFlowSpanish
   ├── data
   │   ├── development.tsv
   │   ├── train.tsv
   │   └── test.tsv
   ├── features
   │   ├── xxxxxxxxx
   │   └── xxxxxxxxx
   └── inputs
       ├── xxxxxxxxx
       └── xxxxxxxxx
```

### Installation

To generate the features and run the model, we recommend the following running the scripts on python 3.6 and the following requirements (following the original paper). More current versions of the libraries raise errors. If you have access to a GPU, use `tensorflow-gpu` instead of `tensorflow`.
```
gensim==3.8.0
tensorflow==1.14.0
pandas==0.24.2
nltk==3.4.5
scikit-learn==0.20.2
tqdm==4.32.1
keras==2.2.4
Keras-Preprocessing==1.1.0
keras-self-attention==0.35.0
h5py==2.10.0
numpy==1.16.4
spacy==3.6.1
```
Download NLTK punctuation:
```
$ python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
```
Download a SpaCy model:
exit from python environment first, then write in the prompt:
```
python -m spacy download es_core_news_sm
```

### Reproducing the experiments

First, you need to run the `process_resources.py` script, to process the resources that you will have downloaded into the correct format and into the correct location:
```python
python process_resources.py
```

To train the model, run the `main.py` script as follows:
```python
python main.py [parameters]
```

For example:
```python
python main.py -d fakedes -o outputs/both_branches.check -of 1 -ep embeddings/data --mode traintest -ub both_branches
```

This is the list of parameters the user can pass as input:

##### General parameters

| parameter          | flag | description                                                 | default value   | required |
|--------------------|------|-------------------------------------------------------------|-----------------|----------|
| dataset            | d    | dataset name (str)                                          | `fakedes`       | no       |
| segments_number    | sn   | number of segments (int)                                    | `10`              | no       |
| overwrite_features | of   | overwrite features or not: 0 or 1 (int)                     | `0`               | no       |
| mode               | m    | `train`, `test`, or `apply` (str)                           | `train`         | no       |
| use_branches       | ub   | `both_branches`, `affective_branch` or `topic_branch` (str) | `both_branches` | no       |
| search             | s    | search for best parameters: 0 or 1 (int)                    | `0`               | no       |
| output_dir         | o    | output directory for the model (str)                        |                 | yes      |

##### Network parameters
| parameter          | flag | description                                                 | default value   | required |
|--------------------|------|-------------------------------------------------------------|-----------------|----------|
| embedding_path     | ep   | path to the word2vec embeddings file (str) | `./my_embeddings.vec` | no
| embedding_size     | es   | size of the word2vec embeddings (int) | `300` | no
| rnn_size     | rs   | size of recurrent neural network (int) | `8` | no
| num_filters     | nf   | number of filters (int) | `16` | no
| filter_sizes     | fs   |sizes of the filters (int) | `[2,3,4]` | no
| activation_rnn     | ar   |activation of the recurrent neural network (str) | `tanh` | no
| pool_size     | ps   |the size of the pool (int) | `3` | no
| activation_cnn     | ac   |activation of the convolutional neural network (str) | `relu` | no
| dense_1     | d1 | first dense layer (int) | `8` | no
| activation_attention     | aa | defines the kind of activation attention used (str) | `softmax` | no
| dense_2     | d2 | second dense layer (int) | `8` | no
| dense_3     | d3 | third dense layer (int) | `8` | no
| dropout     | dt | dimensions of the dropout (int) | `0.3910` | no
| optimizer    | op | kind of optimizer used (str) | `adam` | no
| max_senten_len    | ml | the maximum lenght of a sentence (int) | `500` | no
| vocab    | vb | vocab (int) | `1000000` | no
| max_epoch    | me | the maximum numbers of epochs (int) | `50` | no
| batch_size    | bs | the size of the batch (int) | `16` | no
### Credit and citation

This code is taken and adapted from [FakeFlow](https://github.com/bilalghanem/fake_flow), the code accompanying the following paper:
```
@inproceedings{ghanem2021fakeflow,
  title={{FakeFlow: Fake News Detection by Modeling the Flow of Affective Information}},
  author={Ghanem, Bilal and Ponzetto, Simone Paolo and Rosso, Paolo and Rangel, Francisco},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2021}
}
```
Please make sure to refer to the original paper or code repository if you use either of them.

If you use the Spanish version of FakeFlow, please cite the following paper:
```
@inproceedings{togni2024fakeflow,
  title={Emotions and News Structure: An Analysis of the Language of Fake News in Spanish},
  author={Togni, Benedetta and Coll Ardanuy, Mariona and Chulvi, Berta and Rosso, Paolo},
  booktitle={SEPLN-2024: 40th Conference of the Spanish Society for Natural Language Processing: posters},
  year={2024}
}
```