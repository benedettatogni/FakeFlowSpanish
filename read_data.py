import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from features.building_features import segmentation_text, clean_regex, my_selected_features
from sklearn.utils import shuffle

import spacy
nlp = spacy.load("es_core_news_sm")

import random
np.random.seed(0)
random.seed(0)

def my_prepare_input(
        dataset='MyData', 
        segments_number=10, 
        text_segments=False, 
        clean_text=True, 
        overwrite=False
        ):

    # Load dataset:
    train = pd.read_csv('./data/{}/train.tsv'.format(dataset), sep="\t")
    dev = pd.read_csv('./data/{}/development.tsv'.format(dataset), sep="\t")
    test = pd.read_csv('./data/{}/test.tsv'.format(dataset), sep="\t")
    # Rename test set columns to match train and dev dataframe:
    test = test.rename(columns={"ID": "Id",
                                "CATEGORY": "Category",
                                "TOPICS": "Topic",
                                "SOURCE": "Source",
                                "HEADLINE": "Headline",
                                "TEXT": "Text",
                                "LINK": "Link"})
    # Add empty column to match train and dev dataframes:
    test["Link"] = ""
    
    original_dataset = {"train": train,
                        "dev": dev,
                        "test": test}
    processed_dataset = []

    # Process each data split:
    for data_split in ["train", "dev", "test"]:
        tmp_df = original_dataset[data_split]
        # Unify category labels:
        tmp_df["Category"] = tmp_df["Category"].map({"Fake": 1, "True": 0, "FALSE": 1, "TRUE": 0, "False": 1, True: 0, False: 1})
        # Drop empty lines and remove multiple whitespaces:
        texts = tmp_df['Text'].to_list()
        content = [texts[i].replace("\n", " ") for i in range(len(texts))]
        content = [" ".join(x.split()) for x in content]
        # Get lemmatised content for features:
        lemmatised_content = [" ".join([w.lemma_ for w in nlp(x) if w.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]) for x in content]
        # lemmatised_content = content
        # # Extract features, segment text and clean it:
        # content_features = my_manual_features(
        #     path='./features', 
        #     model_name=dataset,
        #     data_split=data_split,
        #     segments_number=segments_number,
        #     overwrite=overwrite
        #     ).transform(lemmatised_content)
        # Extract features, segment text and clean it:
        content_features = my_selected_features(
            path='./features', 
            model_name=dataset,
            data_split=data_split,
            segments_number=segments_number,
            overwrite=overwrite
            ).transform(lemmatised_content)
        
        # In segmentation we already clean the text to keep the DOTS (.) only:
        if text_segments:
            tmp_df["content"] = segmentation_text(segments_number=segments_number).transform(content)
        elif clean_text:
            tmp_df["content"] = content.map(lambda text: clean_regex(text, keep_dot=True))

        # Shuffle df and rename columns:
        tmp_df["features"] = content_features.tolist()
        tmp_df = shuffle(tmp_df, random_state=0)
        tmp_df = tmp_df.reset_index(drop=True).reset_index()
        tmp_df = tmp_df.rename(columns={
                                "content": "text",
                                "Id": "id",
                                "Category": "label",
                                "Topic": "topic"})
        # Keep only columns of interest:
        tmp_df = tmp_df.drop(columns=["index", "Source", "Headline", "Text", "Link"])
        # Store dataframe:
        tmp_df.to_csv("./processed_files/{}_df.tsv".format(data_split), sep="\t", index=False)
        # Append final dataframes into list:
        processed_dataset.append(tmp_df)

    return processed_dataset

# def my_prepare_plot_attn_input(dataset='MyData', segments_number=10, n_jobs=-1, emo_rep='frequency', return_features=True,
#                   text_segments=False, clean_text=True):

#     content = pd.read_csv('./data/{}/sample.csv'.format(dataset))
#     content_features = []
#     """Extract features, segment text, clean it."""
#     if return_features:
#         content_features = my_manual_features(n_jobs=n_jobs, path='./features', model_name=dataset,
#                                            segments_number=segments_number, emo_rep=emo_rep).transform(content['content'])

#     print("\n----> AFTER my_manual_features:", content_features.shape, '\n')
#     """In segmentation we already clean the text to keep the DOTS (.) only."""
#     if text_segments:
#         content['content'] = segmentation_text(segments_number=segments_number).transform(content['content'])
#     elif clean_text:
#         content['content'] = content['content'].map(lambda text: clean_regex(text, keep_dot=True))

#     train, dev, test = split(content, content_features, return_features)
#     test['raw_text'] = content[content['type'] == 'test']['content'].tolist()
#     test['raw_label'] = content[content['type'] == 'test']['label'].tolist()
#     return train, dev, test


if __name__ == '__main__':
    train, dev, test = my_prepare_input(dataset='MultiSourceFake', segments_number=10, text_segments=True)