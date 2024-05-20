import pandas as pd

import spacy
nlp = spacy.load("es_core_news_sm")

print("Emotions lexicon")
df1 = pd.read_csv("./features/emotional/Spanish-NRC-EmoLex.txt", sep='\t', index_col = 0)
df1 = df1.rename(columns={"Spanish Word": "word"})
# Drop multi-word expressions:
df1 = df1[~df1["word"].str.contains(" ")]
# Keep only the following columns:
df1 = df1[["word", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]]
# Lemmatize with SpaCy:
df1["word"] = df1["word"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
# Drop multi-word expressions:
df1 = df1[~df1["word"].str.contains(" ")]
# Lower-case words:
df1["word"] = df1["word"].str.lower()
# Remove duplicates:
df1 = df1.drop_duplicates()
df1.to_csv("./features/emotional/emotions_resource.txt", sep="\t", index=False)

print("Sentiments lexicon")
df = pd.read_csv("./features/emotional/Spanish-NRC-EmoLex.txt", sep='\t', index_col = 0)
df = df.rename(columns={"Spanish Word": "word"})
# Drop multi-word expressions:
df = df[~df["word"].str.contains(" ")]
# Keep only the following columns:
df = df[["word", "positive", "negative"]]
# df["word"] = df["word"].apply(lambda x: stemmer.stem(x))
# df = df[df["word"].str.len() >= 4]
# Lemmatize with SpaCy:
df["word"] = df["word"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
# Remove duplicates:
df = df.drop_duplicates()
df.to_csv("./features/sentiment/sentiment_resource.txt", sep="\t", index=False)

# Process the Guasch et al. (2015) lexicon:
print("Imageability lexicon")
df = pd.read_csv("./features/imageability/imageability_es.csv")
list_rows = []
# Keep only the following columns:
df = df[["Word", "VAL_M", "ARO_M", "CON_M", "IMA_M"]]
# Rename columns:
df = df.rename(columns={"Word": "word", "VAL_M": "valence", "ARO_M": "arousal", "CON_M": "concreteness", "IMA_M": "imageability"})
# df["word"] = df["word"].apply(lambda x: stemmer.stem(x))
# df = df[df["word"].str.len() >= 4]
# Lemmatize with SpaCy:
df["word"] = df["word"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
# Remove duplicates:
df = df.drop_duplicates()
# Normalize scores:
df["valence"] = df["valence"] / df["valence"].max() # Valence
df["arousal"] = df["arousal"] / df["arousal"].max() # Arousal
df["concreteness"] = df["concreteness"] / df["concreteness"].max() # Concreteness
df["imageability"] = df["imageability"] / df["imageability"].max() # Imageability
df.to_csv("./features/imageability/imageability_resource.tsv", sep="\t", index=False)

# Process the HurtLex lexicon:
print("Hurtful lexicon")
df = pd.read_csv("./features/hurtful/hurtlex_ES.tsv", sep="\t")
# Lower-case the lemma
df["lemma"] = df["lemma"].str.lower()
# Keep only words in the following categories:
hurtlex_words = df[df["category"].isin(["ps", "ddf", "ddp", "dmc", "is", "pr", "om", "qas", "cds", "re", "svp"])]["lemma"].to_list()
# # Lemmatize with SpaCy:
hurtlex_words = [" ".join([w.lemma_ for w in nlp(x)]) for x in hurtlex_words]
# df["lemma"] = df["lemma"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
# Remove duplicates:
hurtlex_words = list(set(hurtlex_words))
# Remove words containing whitespaces:
hurtlex_words = [x for x in hurtlex_words if not " " in x]
# # Stem words:
# hurtlex_words = [stemmer.stem(x) for x in hurtlex_words]
# # Minimum length of 4:
# hurtlex_words = [x for x in hurtlex_words if len(x) >= 4]
# Lemmatize with SpaCy:
# Store resource:
with open("./features/hurtful/hurtful_resource.txt", "w") as f:
    for word in hurtlex_words:
        f.write(f"{word.strip()}\n")

# Process the hyperbolic resource:
print("Hyperbolic lexicon")
hypwords = []
with open("./features/hyperbolic/hyperbolic_es.txt") as fr:
    words = fr.readlines()
    for word in words:
        word = word.strip()
        # # Lemmatize with SpaCy:
        word = " ".join([w.lemma_ for w in nlp(word)])
        # word = stemmer.stem(word)
        if not " " in word:# and len(word) >= 4:
            hypwords.append(word)
hypwords = list(set(hypwords))
# Store resource:
with open("./features/hyperbolic/hyperbolic_resource.txt", "w") as f:
    for word in hypwords:
        f.write(f"{word.strip()}\n")