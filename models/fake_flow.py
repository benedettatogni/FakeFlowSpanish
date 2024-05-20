#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os, random, gensim
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import pandas as pd
from nltk import tokenize
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Keras
import keras
from keras import backend as K
from keras.layers import Dense, GRU, Embedding, Input, Dropout, Bidirectional, MaxPooling1D, Convolution1D, Flatten, Concatenate, concatenate, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras_self_attention import SeqSelfAttention

seed = 0
import random
import numpy as np
import tensorflow as tf
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)

config = tf.ConfigProto(device_count={'CPU': 1})

class FakeFlow:

    def __init__(self, parameters, SHUFFLE=False):
        # Settings
        self.scoring = 'f1'
        self.verbose = 1
        self.SHUFFLE = SHUFFLE
        self.model_name = '{}_{}'.format(parameters['dataset'], parameters['segments_number'])
        self.summary_table = {}
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(['a'])

        self.parameters = parameters
        self.parameters['max_senten_num'] = parameters['segments_number']
        self.tokenizer = Tokenizer(num_words=self.parameters['vocab'], oov_token=True)

    def shuffle_along_axis(self, a):
        idx = [item for item in range(a['features'].shape[1])]
        idx = shuffle(idx, random_state=0)
        a['features'] = a['features'][:, idx, :]
        a['text'] = np.array([item.split('.') for item in a['text']])
        a['text'] = a['text'][:, idx]
        a['text'] = np.array([' . '.join(item) for item in a['text'].tolist()])
        return a

    def prepare_fake_flow_input(self, train, dev, test):
        if self.SHUFFLE == True:
            train = self.shuffle_along_axis(train)
            dev = self.shuffle_along_axis(dev)
            test = self.shuffle_along_axis(test)
            self.model_name += '_shuffled'

        # -- reading tweet text
        self.train, self.dev, self.test = train, dev, test
        all_text = pd.concat([self.train["text"], self.dev["text"], self.test["text"]]).to_list()

        # -- tokenizing and preprocessing input text
        self.tokenizer.fit_on_texts(all_text)
        self.parameters['vocab'] = len(self.tokenizer.word_counts) + 1

        self.train['text'], self.train['label'] = self.preprocessing(train['text'].to_list(), train['label'].to_list())
        self.dev['text'], self.dev['label'] = self.preprocessing(dev['text'].to_list(), dev['label'].to_list())
        self.test['text'], self.test['label'] = self.preprocessing(test['text'].to_list(), test['label'].to_list())

        print("\n----> AFTER prepare_input:", np.array(self.train['text'].tolist()).shape, '\n')
        
        # -- computing embeddings
        self.my_prep_embed()

    def preprocessing(self, text, label):
        """Preprocessing of the text to make it more resonant for training
        """
        paras = []
        max_sent_num = 0
        max_sent_len = 0
        for idx in tqdm(range(len(text)), desc='Tokenizing text'):
            sentences = tokenize.sent_tokenize(text[idx])
            if len(sentences) > 45:
                print()
            if len(sentences) > max_sent_num:
                max_sent_num = len(sentences)
            tmp = [len(sent.split()) for sent in sentences]
            sent_len = max(tmp)
            if sent_len > max_sent_len:
                max_sent_len = sent_len
            paras.append(sentences)

        data = np.zeros((len(text), self.parameters['max_senten_num'], self.parameters['max_senten_len']), dtype='int32')
        for i, sentences in tqdm(enumerate(paras), desc='Preparing input matrix'):
            for j, sent in enumerate(sentences):
                if j < self.parameters['max_senten_num']:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < self.parameters['max_senten_len'] and word in self.tokenizer.word_index and self.tokenizer.word_index[word] < self.parameters['vocab']:
                            data[i, j, k] = self.tokenizer.word_index[word]
                            k = k+1
                            
        labels = self.preparing_labels(label)
        return data.tolist(), labels.tolist()

    def preparing_labels(self, y):
        y = np.array(y)
        y = y.astype(str)
        if y.dtype.type == np.array(['a']).dtype.type:
            if len(self.labelencoder.classes_) < 2:
                self.labelencoder.fit(y)
                self.Labels = self.labelencoder.classes_.tolist()
            y = self.labelencoder.transform(y)
        labels = to_categorical(y, len(self.Labels))
        return labels

    def my_prep_embed(self):
        print("--------> USING MY EMBEDDINGS!")

        path = './processed_files'
        filename = f'embeddings_for_{self.parameters["dataset"]}.npy'
        file = exists(join(path, filename))
        if file:
            embedding_matrix = np.load(join(path, filename))
        else:
            embedding_path = self.parameters['embedding_path']
            print("---> Loading word2vec file:", embedding_path)
            embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)
            embedding_matrix = np.zeros((self.parameters['vocab'], self.parameters['embedding_size']))

            print("---> Generating dataset word2vec file:", embedding_path)
            for word, i in tqdm(self.tokenizer.word_index.items()):
                if i >= self.parameters['vocab']:
                    continue
                if word in embeddings_index:  # .vocab
                    embedding_matrix[i] = embeddings_index[word]
            embeddings_index = {}
            np.save(join(path, filename), embedding_matrix)

        self.embedding_matrix = embedding_matrix
        print(self.embedding_matrix.shape)

    def network(self, use_branches='both_branches'):
        Embed = Embedding(input_dim=self.parameters['vocab'],
                               output_dim=self.parameters['embedding_size'],
                               input_length=self.parameters['max_senten_len'],
                               trainable=True,
                               weights=[self.embedding_matrix],
                               name='Embed_Layer')

        # -- affective information branch 
        # preprocessing layers are part of the model
        inp_features = Input(shape=(np.array(self.train['features'].to_list()).shape[1], np.array(self.train['features'].to_list()).shape[2]), name='features_input')
        flow_features = Bidirectional(GRU(self.parameters['rnn_size'], activation=self.parameters['activation_rnn'], return_sequences=True, name='rnn'))(inp_features)
        features = Model(inp_features, flow_features)
        if self.verbose == 1:
            print("\n\n Affective Information Branch (Lexicon-based Features)")
            features.summary()

        # -- topic information branch
        # pre processing layers part of the model
        word_input = Input(shape=(self.parameters['max_senten_len'],), dtype='float32')
        z = Embed(word_input)
        conv_blocks = []
        for sz in self.parameters['filter_sizes']:
            conv = Convolution1D(filters=self.parameters['num_filters'], kernel_size=sz, padding="valid", activation=self.parameters['activation_cnn'], strides=1)(z)
            conv = MaxPooling1D(pool_size=self.parameters['pool_size'])(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        wordEncoder = Model(word_input, z)
        if self.verbose == 1:
            print("\n\n Topic Information Branch (Word Embeddings)")
            wordEncoder.summary()

        # Sentences Concated
        # preposs layers are part of the model 
        sent_input = Input(shape=(np.array(self.train["text"].to_list()).shape[1], self.parameters['max_senten_len']), dtype='float32', name='input_2')
        y = TimeDistributed(wordEncoder, name='input_sent2')(sent_input)
        y = Dense(self.parameters['dense_1'], name='dense_1')(y)
        y = concatenate([inp_features, y], axis=2)
        y = Dense(self.parameters['dense_2'], name='dense_2')(y)
        #attention matrix, that is supposed to be the output
        y, attn = SeqSelfAttention(attention_activation=self.parameters['activation_attention'], return_attention=True, name='Self-Attention')(y)

        if use_branches == 'both_branches':
            y = keras.layers.dot([flow_features, y], axes=[1, 1])
            y = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(y)
        elif use_branches == 'affective_branch':
            y = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(flow_features)
        elif use_branches == 'topic_branch':
            y = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(y)

        y = Dense(self.parameters['dense_3'], name='dense_3')(y)
        y = Dropout(self.parameters['dropout'])(y)
        y = Dense(len(self.Labels), activation='softmax', name='final_softmax')(y)
        model = Model(inputs=[inp_features, sent_input], outputs=[y, attn])
        if self.verbose == 1:
            model.summary()
        return model

    def apply_on_dev(self):
        dict_dev = {"id": self.dev["id"].tolist()}
        dev_apply_df = pd.DataFrame.from_dict(dict_dev)
        Y_dev_pred, Y_dev_attn = self.model.predict(
            [np.array(self.dev['features'].to_list()), 
             np.array(self.dev['text'].to_list())
             ], 
             batch_size=self.parameters['batch_size'], verbose=0
             )
        predictions_tuple = list(map(tuple, Y_dev_pred))
        dev_apply_df["feature_scores"] = np.round(
            np.array(self.dev['features'].to_list()), 3).tolist()
        dev_apply_df["conf_true"] = [round(float(x[0]), 3) for x in predictions_tuple]
        dev_apply_df["conf_fake"] = [round(float(x[1]), 3) for x in predictions_tuple]
        Y_dev_pred = np.argmax(Y_dev_pred, axis=1)
        dev_apply_df["prediction"] = list(Y_dev_pred)
        Y_dev = np.argmax(np.array(self.dev['label'].to_list()), axis=1)
        dev_apply_df["label"] = list(Y_dev)
        dev_apply_df.to_csv("data/fakedes/results_classifier_dev.csv", index=False)

    def apply_on_test(self):
        dict_test = {"id": self.test["id"].tolist()}
        test_apply_df = pd.DataFrame.from_dict(dict_test)
        Y_test_pred, Y_test_attn = self.model.predict(
            [np.array(self.test['features'].to_list()), 
             np.array(self.test['text'].to_list())
             ], 
             batch_size=self.parameters['batch_size'], verbose=0
             )
        predictions_tuple = list(map(tuple, Y_test_pred))
        test_apply_df["feature_scores"] = np.round(
            np.array(self.test['features'].to_list()), 3).tolist()
        attn_scores = []
        for attn in Y_test_attn:
            attn_scores.append(attn.tolist())
        test_apply_df["attention_scores"] = attn_scores
        test_apply_df["conf_true"] = [round(float(x[0]), 3) for x in predictions_tuple]
        test_apply_df["conf_fake"] = [round(float(x[1]), 3) for x in predictions_tuple]
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        test_apply_df["prediction"] = list(Y_test_pred)
        Y_test = np.argmax(np.array(self.test['label'].to_list()), axis=1)
        test_apply_df["label"] = list(Y_test)
        test_apply_df.to_csv("data/fakedes/results_classifier_test.csv", index=False)

    def evaluate_on_test(self):
        Y_test_pred, Y_test_attn = self.model.predict(
            [np.array(self.test['features'].to_list()), 
             np.array(self.test['text'].to_list())
             ], 
             batch_size=self.parameters['batch_size'], verbose=0
             )
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(np.array(self.test['label'].to_list()), axis=1)
        print("Test")
        print(classification_report(Y_test, Y_test_pred))
        print("Fake F1 score:", f1_score(Y_test, Y_test_pred, average="binary", pos_label=1))
        print("True F1 score:", f1_score(Y_test, Y_test_pred, average="binary", pos_label=0))
        print("Macro F1 score:", f1_score(Y_test, Y_test_pred, average="macro"))
        print("Accuracy score:", accuracy_score(Y_test, Y_test_pred))

    def evaluate_on_dev(self):
        Y_dev_pred, Y_dev_attn = self.model.predict(
            [np.array(self.dev['features'].to_list()), 
             np.array(self.dev['text'].to_list())
             ],
             batch_size=self.parameters['batch_size'], verbose=0
             )
        Y_dev_pred = np.argmax(Y_dev_pred, axis=1)
        Y_dev = np.argmax(np.array(self.dev['label'].to_list()), axis=1)
        print(classification_report(Y_dev, Y_dev_pred))

    def find_best_threshold(self):
        df = pd.read_csv("data/fakedes/results_classifier_dev.csv")
        list_thresholds = df["conf_fake"].unique().tolist()
        min_threshold = min(list_thresholds)
        max_threshold = max(list_thresholds)
        list_thresholds = np.arange(min_threshold, max_threshold, 0.001)
        # Find best threshold on dev:
        best_threshold = 0.0
        best_score = 0.0
        for threshold in list_thresholds:
            predictions = [1 if x >= threshold else 0 for x in df["conf_fake"]]
            labels = df["label"].values.tolist()
            tmp_f1_score = f1_score(labels, predictions, average="macro")
            if tmp_f1_score > best_score:
                best_threshold = threshold
                best_score = tmp_f1_score
        return best_threshold

    def apply_best_threshold(self, threshold):
        df = pd.read_csv("data/fakedes/results_classifier_test.csv")
        # Apply threshold on test:
        predictions = [1 if x >= threshold else 0 for x in df["conf_fake"]]
        labels = df["label"].values.tolist()
        print("Test with threshold")
        print(classification_report(labels, predictions))
        print("Fake F1 score:", f1_score(labels, predictions, average="binary", pos_label=1))
        print("True F1 score:", f1_score(labels, predictions, average="binary", pos_label=0))
        print("Macro F1 score:", f1_score(labels, predictions, average="macro"))
        print("Accuracy score:", accuracy_score(labels, predictions))
        print("Best threshold:", threshold)

    def run_model(self, model_path, type_='train', use_branches='both_branches'):
        # -- creating the directory in case it doesn't exist
        model_dir = "/".join(model_path.split("/")[:-1])
        os.makedirs(model_dir, exist_ok = True)

        # 1, Compile the model
        self.session = tf.InteractiveSession(config=config)
        self.model = self.network(use_branches)
        self.model.compile(optimizer=self.parameters['optimizer'], loss=['categorical_crossentropy', None], metrics=['accuracy'])

        # 2, Prep
        callback = [EarlyStopping(min_delta=0.0001, patience=4, verbose=2, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.03, patience=3, verbose=2, min_lr=0.00001)]
        callback.append(ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False))

        # 3, Train
        if type_ == 'train':
            self.model.fit(
                x=[np.array(self.train['features'].to_list()), np.array(self.train['text'].to_list())], 
                y=np.array(self.train['label'].to_list()), 
                batch_size=self.parameters['batch_size'], 
                epochs=self.parameters['max_epoch'], 
                verbose=self.verbose,
                validation_data=(
                    [np.array(self.dev['features'].to_list()), 
                     np.array(self.dev['text'].to_list())], 
                     np.array(self.dev['label'].to_list())
                     ), 
                callbacks=callback)
            self.evaluate_on_dev()
            self.apply_on_dev()
        elif type_ == 'test':
            if os.path.exists(model_path):
                print('--------Model exists: ' + model_path)
                self.model.load_weights(model_path, by_name=True)
                print('--------Load Weights Successful!--------')
            else:
                print('Model does not exist: {}'.format(model_path))
                exit(1)
            self.evaluate_on_test()
        elif type_ == 'apply':
            if os.path.exists(model_path):
                print('--------Model exists--------')
                self.model.load_weights(model_path, by_name=True)
                print('--------Load Weights Successful!--------')
            else:
                print('Model does not exist: {}'.format(model_path))
                exit(1)
            self.evaluate_on_test()
            self.apply_on_test()
        elif type_ == 'traintest':
            # Train model:
            self.model.fit(
                x=[np.array(self.train['features'].to_list()), np.array(self.train['text'].to_list())], 
                y=np.array(self.train['label'].to_list()), 
                batch_size=self.parameters['batch_size'], 
                epochs=self.parameters['max_epoch'], 
                verbose=self.verbose,
                validation_data=(
                    [np.array(self.dev['features'].to_list()), 
                     np.array(self.dev['text'].to_list())], 
                     np.array(self.dev['label'].to_list())
                     ), 
                callbacks=callback)
            self.evaluate_on_dev()
            self.apply_on_dev()
            # Find best threshold on dev:
            best_threshold = self.find_best_threshold()
            # Test model:
            if os.path.exists(model_path):
                print('--------Model exists: ' + model_path)
                self.model.load_weights(model_path, by_name=True)
                print('--------Load Weights Successful!--------')
            else:
                print('Model does not exist: {}'.format(model_path))
                exit(1)
            self.evaluate_on_test()
            self.apply_on_test()
            # Apply threshold on test:
            self.apply_best_threshold(best_threshold)
        else:
            print('Mode not defined!')
            exit(1)

        self.session.close()