import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from os.path import join

class FeaturesLexicon:

    def __init__(self, path):
        self.lexicons_path = path
        # Emotions:
        self.nrc = pd.read_csv(join(self.lexicons_path, "emotional/emotions_resource.txt"), sep='\t')
        self.anger = dict(zip(self.nrc["word"], self.nrc["anger"]))
        self.anticipation = dict(zip(self.nrc["word"], self.nrc["anticipation"]))
        self.disgust = dict(zip(self.nrc["word"], self.nrc["disgust"]))
        self.fear = dict(zip(self.nrc["word"], self.nrc["fear"]))
        self.joy = dict(zip(self.nrc["word"], self.nrc["joy"]))
        self.sadness = dict(zip(self.nrc["word"], self.nrc["sadness"]))
        self.surprise = dict(zip(self.nrc["word"], self.nrc["surprise"]))
        self.trust = dict(zip(self.nrc["word"], self.nrc["trust"]))
        # Sentiment:
        self.nrc = pd.read_csv(join(path, 'sentiment/sentiment_resource.txt'), sep='\t')
        self.positive = self.nrc[self.nrc['positive'] == 1]['word'].tolist()
        self.negative = self.nrc[self.nrc['negative'] == 1]['word'].tolist()
        # Imageability:
        self.reader = pd.read_csv(join(path, "imageability/imageability_resource.tsv"), sep="\t")
        self.valence = dict(zip(self.reader["word"], self.reader["valence"]))
        self.arousal = dict(zip(self.reader["word"], self.reader["arousal"]))
        self.concreteness = dict(zip(self.reader["word"], self.reader["concreteness"]))
        self.imageability = dict(zip(self.reader["word"], self.reader["imageability"]))
        # Hyperbolic:
        self.reader = pd.read_csv(join(path, 'hyperbolic/hyperbolic_resource.txt'), sep='\n', names=["word"])
        self.hyper = self.reader['word'].tolist()
        # Hurtful:
        self.reader = pd.read_csv(join(path, 'hurtful/hurtful_resource.txt'), sep='\n', names=["word"])
        self.hurtful = self.reader['word'].tolist()

    def frequency(self, sentence):
        words = []
        for word in sentence:
            try:
                anger = self.anger.get(word, 0.0)
                anticipation = self.anticipation.get(word, 0.0)
                disgust = self.disgust.get(word, 0.0)
                fear = self.fear.get(word, 0.0)
                joy = self.joy.get(word, 0.0)
                sadness = self.sadness.get(word, 0.0)
                surprise = self.surprise.get(word, 0.0)
                trust = self.trust.get(word, 0.0)
                pos = 1 if word in self.positive else 0
                neg = 1 if word in self.negative else 0
                valence = self.valence.get(word, 0.0)
                arousal = self.arousal.get(word, 0.0)
                concreteness = self.concreteness.get(word, 0.0)
                imageability = self.imageability.get(word, 0.0)
                hyper = 1 if word in self.hyper else 0
                hurtful = 1 if word in self.hurtful else 0
                # All features:
                result = [anger, anticipation, disgust, fear, joy, sadness, surprise, trust, pos, neg, valence, arousal, concreteness, imageability, hyper, hurtful]
                # # Selected features (for the ablation study):
                # result = [anger, anticipation, disgust, fear, joy, sadness, surprise, trust, pos, neg, valence, arousal, concreteness, imageability, hyper]
            except:
                # All features:
                result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # # Selected_features (for the ablation study):
                # result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            words.append(result)

        if len(words) == 0:
            # All features:
            words.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # # Selected_features (for the ablation study):
            # words.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return words

   
