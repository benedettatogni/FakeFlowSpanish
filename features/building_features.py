import warnings
warnings.filterwarnings("ignore")

import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tqdm import tqdm
from os.path import exists
from os.path import join
from features.all_features.loading_selected_lexicon import FeaturesLexicon
from pathlib import Path

# config
np.random.seed(0)
tqdm.pandas()

def clean_regex(text, keep_dot=False, split_text=False):
    try:
        text = re.sub(r'((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=;%&:/~+#-]*[\w@?^=%&;:/~+#-])?)', ' ',
                      text)
        text = re.sub(r'[^ ]+\.com', ' ', text)
        text = re.sub(r'(\d{1,},)?\d{1,}(\.\d{1,})?', '', text)
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñü\'. ]', ' ', text)
        text = re.sub(r'\.', '. ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        text = re.sub(r'(\.\s)+', '.', str(text).strip())
        text = re.sub(r'\.{2,}', '.', str(text).strip())
        text = re.sub(r'(?<!\w)([A-Z])\.', r'\1', text)

        text = re.sub(r'\'(?!\w{1,2}\s)', ' ', text)

        text = text.split('.')
        if keep_dot:
            text = ' '.join([sent.strip() + ' . ' for sent in text])
        else:
            text = ' '.join([sent.strip() for sent in text])

        text = text.lower()
        return text.split() if split_text else text
    except:
        text = 'empty text'
        return text.split() if split_text else text

class append_split_3D(BaseEstimator, TransformerMixin):
    def __init__(self, segments_number=10, max_len=50, mode='append'):
        self.segments_number = segments_number
        self.max_len = max_len
        self.mode = mode
        self.appending_value = -5.123

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.mode == 'append':
            self.max_len = self.max_len - data.shape[2]
            appending = np.full((data.shape[0], data.shape[1], self.max_len), self.appending_value)
            new = np.concatenate([data, appending], axis=2)
            return new
        elif self.mode == 'split':
            tmp = []
            for item in range(0, data.shape[1], self.segments_number):
                tmp.append(data[:, item:(item + self.segments_number), :])
            tmp = [item[item != self.appending_value].reshape(data.shape[0], self.segments_number, -1) for item in tmp]
            new = np.concatenate(tmp, axis=2)
            return new
        else:
            print('Error: Mode value is not defined')
            exit(1)

class segmentation(BaseEstimator, TransformerMixin):

    def __init__(self, segments_number=10):
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        out = []
        for sentence in data:
            tmp = np.array_split(sentence, self.segments_number)
            tmp = [np.sum(item, axis=0) / sentence.shape[0] for item in tmp]
            out.append(tmp)
        out = np.array(out)
        return out

class segmentation_text(BaseEstimator, TransformerMixin):

    def __init__(self, segments_number=20):
        self.segments_number = segments_number

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        data = [clean_regex(sentence, keep_dot=False, split_text=True) for sentence in tqdm(data, desc='Text Segmentation')]
        if isinstance(data, list):
            data = np.array([np.array(sent) for sent in data])
        out = []
        for sentence in data:
            try:
                tmp = np.array_split(sentence, self.segments_number)
                tmp = ' . '.join([' '.join(item.tolist()) for item in tmp])
            except:
                print()
            out.append(tmp)
        return out

class selected_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', model_name='', overwrite=0, data_split=None):
        self.path = path
        self.model_name = model_name
        self.overwrite = overwrite
        self.data_split = data_split

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        file_name = './processed_files/features/selected_features_{}_{}.npy'.format(self.model_name, self.data_split)
        if exists(file_name) and not self.overwrite:
            print(f"\n-----> Loading features from {file_name}")
            features = np.load(file_name, allow_pickle=True).tolist()
        else:
            Path("processed_files/features/").mkdir(parents=True, exist_ok=True)
            print(f"\n-----> Creating the features {file_name}")
            # Clean the data:
            data = [clean_regex(sentence, False, True) for sentence in tqdm(data, desc='Cleaning text')]
            # Generate lexicon:
            fl = FeaturesLexicon(self.path)
            loop = tqdm(data)
            loop.set_description('Building selected_features')
            # Generate selected features:
            features = [fl.frequency(sentence) for sentence in loop]
            features = [np.array(item) for item in features]
            np.save(file_name, features)
        return features

def my_selected_features(path='', model_name='', data_split=None, segments_number=10, overwrite=0):
    manual_feats = Pipeline([
        ('FeatureUnion', FeatureUnion([
            ('1', Pipeline([
                ('selected_features', selected_features(path=path, model_name=model_name, overwrite=overwrite, data_split=data_split)),
                ('segmentation', segmentation(segments_number=segments_number)),
                ('append', append_split_3D(segments_number=segments_number, max_len=50, mode='append')),
            ])),
        ])),
        ('split', append_split_3D(segments_number=segments_number, max_len=50, mode='split'))
    ])
    return manual_feats