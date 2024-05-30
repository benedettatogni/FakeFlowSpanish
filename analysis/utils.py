import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
import re
from transformers import pipeline

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import spacy
nlp = spacy.load("es_core_news_sm")
ckpt = "Narrativaai/fake-news-detection-spanish"

dict_colors = {"anger": "indianred",
            "anticipation": "lightsalmon",
            "disgust": "palevioletred",
            "fear": "forestgreen",
            "joy": "gold",
            "sadness": "steelblue",
            "surprise": "darkcyan",
            "trust": "palegreen",
            "hurtful": "darkgrey",
            "negative": "dimgrey"}


# ----------------------------------------------------
# Process data
def merge_data_outputs(split, feature_labels):
    dMapColumns = {"Id": "id", "Category": "category", "Topic": "topic", "Source": "source", "Headline": "headline", "Text": "text", "Link": "link",
                   "ID": "id", "CATEGORY": "category", "TOPICS": "topic", "SOURCE": "source", "HEADLINE": "headline", "TEXT": "text", "LINK": "link"}
    dMapLabels = {"Fake": 1, "False": 1, False: 1, "True": 0, True: 0}
    dMapTopics = {"Sociedad": "society",
                "Política": "politics",
                "Ciencia": "science",
                "Ambiental": "environment",
                "Deporte": "sports",
                "Internacional": "international",
                "Sport": "sports",
                "Politics": "politics",
                "Entertainment": "entertainment",
                "Society": "society",
                "Science": "science",
                "Health": "health",
                "Economy": "economy",
                "Security": "security",
                "Education": "education",
                "Covid-19": "covid-19",
                }
    # Read original dataset:
    if split == "dev":
        df_original = pd.read_csv("../data/fakedes/development.tsv", sep="\t")
    else:
        df_original = pd.read_csv("../data/fakedes/" + split + ".tsv", sep="\t")
    # Read dataframe with features:
    df_features = pd.read_csv("../processed_files/" + split + "_df.tsv", sep="\t")
    # Rename columns to homogenise, drop repeated columns:
    df_features = df_features.rename(columns=dMapColumns)
    df_features = df_features.drop(columns=["text", "topic"])
    df_original = df_original.rename(columns=dMapColumns)
    # Merge the two dataframes
    df_merged = pd.merge(df_original, df_features, on="id")
    # Homogenise labels:
    df_merged["category"] = df_merged["category"].astype(str)
    df_merged["category"] = df_merged["category"].replace(dMapLabels)
    # Homogenise topics:
    df_merged["topic"] = df_merged["topic"].replace(dMapTopics)
    # Assert that the gs in both datasets match:
    print(df_merged["category"].to_list() == df_merged["label"].to_list())
    # For the dev and test datasets...:
    if not split == "train":
        # Read the dataframe with the results from the classifier:
        df_results = pd.read_csv("../outputs/results_classifier_" + split + ".csv")
        # Homogenise column names:
        df_results = df_results.rename(columns=dMapColumns)
        # Merge with the original dataset and the features dataframe:
        df_merged = pd.merge(df_merged, df_results, on="id")
        # Assert that the merging has been correct:
        print(df_merged["label_x"].to_list() == df_merged["label_y"].to_list())
        # Drop duplicate columns, homogenise column names:
        df_merged = df_merged.drop(columns=["feature_scores"])
        df_merged = df_merged.rename(columns={"label_y": "label"})
        print(df_merged["label"].to_list() == df_merged["category"].to_list())
        df_merged = df_merged.drop(columns=["label_x"])
    # If the dataset is test...:
    if split == "test": 
        # Prepare the attention scores to be processed:
        df_merged['attention_scores'] = df_merged['attention_scores'].apply(lambda x: literal_eval(str(x)))
        df_merged["attention_scores"] = np.array([x for x in df_merged["attention_scores"]]).round(3).tolist()
    # Prepare the features to be processed:
    df_merged["features"] = df_merged["features"].apply(lambda x: literal_eval(str(x)))
    df_merged["features"] = np.array([x for x in df_merged["features"]]).round(3).tolist()
    # Drop duplicate columns:
    df_merged = df_merged.drop(columns=["category"])
    # Add an average feature values row:
    feature_rows = []
    for i, row in df_merged.iterrows():
        # Average by column (all features into a column, values summed up):
        feature_rows.append(np.array(row["features"]).sum(0).round(4).tolist())
    # Join dataframes:
    features_df_sum = pd.DataFrame(feature_rows, columns=feature_labels)
    print(df_merged.shape[0] == features_df_sum.shape[0])
    df_merged = pd.concat([df_merged, features_df_sum], axis=1)
    return df_merged


def merge_results_df(results_df, original_test_df):
    test_df = pd.merge(original_test_df, results_df, on="id")
    assert_mapping = test_df["label"].to_list() == test_df["gs"].to_list()
    print(assert_mapping)
    test_df = test_df.drop(columns=["gs", "headline"])
    test_df["link"] = test_df["link"].fillna("")
    test_df['attention_scores'] = test_df.attention_scores.apply(lambda x: literal_eval(str(x)))
    test_df['feature_scores'] = test_df.feature_scores.apply(lambda x: literal_eval(str(x)))
    test_df["source"] = test_df["source"].fillna("")
    # test_df = test_df[test_df["source"] != "AFPFactual"]
    return test_df


# ----------------------------------------------------
# Assess performance on test set
def test_performance(test_df):
    y_test = test_df["label"].to_list()
    y_test_pred = test_df["prediction"].to_list()

    print("Fake F1 score:", classification_report(y_test, y_test_pred))
    print("Fake F1 score:", round(f1_score(y_test, y_test_pred, average="binary", pos_label=1), 3))
    print("True F1 score:", round(f1_score(y_test, y_test_pred, average="binary", pos_label=0), 3))
    print("Precision score:", round(precision_score(y_test, y_test_pred, average="macro"), 3))
    print("Recall score:", round(recall_score(y_test, y_test_pred, average="macro"), 3))
    print("Macro F1 score:", round(f1_score(y_test, y_test_pred, average="macro"), 3))
    print("Accuracy score:", round(accuracy_score(y_test, y_test_pred), 3))

# ----------------------------------------------------
# Building a radar chart of the MI gain of each feature
# From: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
# Copyright 2002–2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012-2023 The Matplotlib development team.

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# ----------------------------------------------------
# Building the visualisation

def highlight_emotions(sentence):

    # Emotions:
    nrc = pd.read_csv("../features/emotional/emotions_resource.txt", sep='\t')
    anger = dict(zip(nrc["word"], nrc["anger"]))
    anticipation = dict(zip(nrc["word"], nrc["anticipation"]))
    disgust = dict(zip(nrc["word"], nrc["disgust"]))
    fear = dict(zip(nrc["word"], nrc["fear"]))
    joy = dict(zip(nrc["word"], nrc["joy"]))
    sadness = dict(zip(nrc["word"], nrc["sadness"]))
    surprise = dict(zip(nrc["word"], nrc["surprise"]))
    trust = dict(zip(nrc["word"], nrc["trust"]))
    # Sentiment:
    nrc = pd.read_csv("../features/sentiment/sentiment_resource.txt", sep='\t')
    positive = nrc[nrc['positive'] == 1]['word'].tolist()
    negative = nrc[nrc['negative'] == 1]['word'].tolist()
    # Imageability:
    reader = pd.read_csv("../features/imageability/imageability_resource.tsv", sep="\t")
    valence = dict(zip(reader["word"], reader["valence"]))
    arousal = dict(zip(reader["word"], reader["arousal"]))
    concreteness = dict(zip(reader["word"], reader["concreteness"]))
    imageability = dict(zip(reader["word"], reader["imageability"]))
    # Hyperbolic:
    hyper = []
    with open("../features/hyperbolic/hyperbolic_resource.txt") as f:
        hyper = f.readlines()
    # Hurtful:
    hurtful = []
    with open("../features/hurtful/hurtful_resource.txt") as f:
        hurtful = f.readlines()

    sentence_tokens = []
    for x in sentence:
        sentence_tokens += [(w.text, w.lemma_, w.pos_) for w in nlp(x) if w.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and len(w.text) >= 3]
    emotions_to_highlight = {key: [] for key in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "hurtful", "negative"]}
    dDegrees = dict()
    for (word, lemma, pos) in sentence_tokens:
        if anger.get(lemma):
            emotions_to_highlight["anger"].append(word)
        if anticipation.get(lemma):
            emotions_to_highlight["anticipation"].append(word)
        if disgust.get(lemma):
            emotions_to_highlight["disgust"].append(word)
        if fear.get(lemma):
            emotions_to_highlight["fear"].append(word)
        if joy.get(lemma):
            emotions_to_highlight["joy"].append(word)
        if sadness.get(lemma):
            emotions_to_highlight["sadness"].append(word)
        if surprise.get(lemma):
            emotions_to_highlight["surprise"].append(word)
        if trust.get(lemma):
            emotions_to_highlight["trust"].append(word)
        if lemma in hurtful:
            emotions_to_highlight["hurtful"].append(word)
        if lemma in negative:
            emotions_to_highlight["negative"].append(word)
        if lemma in valence:
            dDegrees[lemma] = [valence[lemma], arousal[lemma], concreteness[lemma], imageability[lemma], pos]
        
    return emotions_to_highlight, dDegrees


def color_emotions(text, emotions):

    emotion_text = {}
    l_keywords = list(emotions[e] for e in emotions)
    l_keywords = list(set([x for xs in l_keywords for x in xs]))
    for k in l_keywords:
        emotions_per_keyword = [e for e in emotions if k in emotions[e]]
        if len(emotions_per_keyword) == 1:
            color = dict_colors[emotions_per_keyword[0]]
            colored_keyword = f'<a style="background:{color}; color:black; padding:3px; width: 90px;" >{k}</a>'
            text = re.sub(r'\b' + re.escape(k) + r'\b', colored_keyword, text, flags=re.IGNORECASE)
            if emotions_per_keyword[0] not in emotion_text:
                emotion_text[emotions_per_keyword[0]] = []
            emotion_text[emotions_per_keyword[0]].append(colored_keyword)
        else:
            percentage = 100/len(emotions_per_keyword)
            percentage_acc = 0
            color_str = ""
            for perc_emotion in emotions_per_keyword:
                color_str += ", " + dict_colors[perc_emotion] + " " + str(int(percentage_acc)) + "% " + str(int(percentage + percentage_acc)) + "%"
                percentage_acc += percentage
            colored_keyword = f'<a style="background: linear-gradient({color_str[2:]}); color: black; padding:3px; width: 90px;">{k}</a>'
            text = re.sub(r'\b' + re.escape(k) + r'\b', colored_keyword, text, flags=re.IGNORECASE)

            for emotion in emotions_per_keyword:
                if emotion not in emotion_text:
                    emotion_text[emotion] = []
            emotion_text[emotion].append(colored_keyword)

    return text

def create_colored_html_with_rectangle(original_text, attn_indices):
    square_colors = ["#DCDCDC", "#D0D0D0", "#BEBEBE", "#A9A9A9", "#989898", "#808080", "#696969", "#585858", "#404040", "#282828"]

    # Emotions:
    sorted_emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "negative", "hurtful"]

    # Select grey color:
    square_colors = [square_colors[x] for x in attn_indices]

    # Imposta una larghezza fissa per testo e legenda
    text_width = 900

    # Imposta una larghezza più piccola per la voce "trust" nella legenda
    highlighted_word_width = 60

    # Imposta la larghezza specifica per la voce "anticipation" nella legenda
    anticipation_width = 90

    # Aumenta la larghezza dei quadrati
    square_width = 20

    # Aggiorna la larghezza per i paragrafi del testo in modo da corrispondere alla larghezza della legenda
    colored_lines = [f'<div style="background-color: {square_colors[i]}; height: 1em; width: {square_width}px; float: left; border: 1px solid gray; margin-right: 10px;"></div><p style="width: {text_width}px;">{line}</p>' for i, line in enumerate(original_text.split('\n')[:5])]

    # Aggiorna la larghezza della voce "trust" e "anticipation" nella legenda, imposta il display inline-block
    legend_content = "".join([f'<span style="margin-right: 10px; background-color:{dict_colors[emotion]}; padding: 3px; width: {highlighted_word_width if emotion != "anticipation" else anticipation_width}px; display: inline-block;"><strong style="color:black;">{emotion.capitalize()}</strong></span>' for emotion in sorted_emotions])

    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .container {{
                max-width: {text_width + 40}px;  /* Imposta la larghezza del contenitore per ospitare legenda e testo con margine */
                text-align: left;
            }}
            .tweet, .emotions, .colored-text {{
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="tweet">
                {''.join(colored_lines)}
            </div>

            <div class="emotions">
                <div style="display: flex; flex-wrap: wrap;">{legend_content}</div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


# -------------------------------------------------
# Apply RoBERTa classifier

def apply_roberta(df_path):

    classifier = pipeline("text-classification", model=ckpt, max_length=512, truncation=True)

    df = pd.read_csv(df_path, sep="\t")
    df["HEADLINE"] = df["HEADLINE"].fillna("")

    predicted_labels = []
    predicted_score = []
    for i, row in df.iterrows():
        text = row["HEADLINE"] + " [SEP] " + row["TEXT"]
        results_roberta = classifier(text)
        predicted_labels.append(results_roberta[0]["label"])
        predicted_score.append(results_roberta[0]["score"])

    df["predicted_label"] = predicted_labels
    df["predicted_score"] = predicted_score

    df = df.drop(columns=["HEADLINE", "TEXT", "LINK"])
    return df
    