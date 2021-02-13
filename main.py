import pandas as pd  # lettura file csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from funzioni import *

# permette di vettorizzare i dati del dataset di training X
from sklearn.feature_extraction.text import CountVectorizer
# tokenizer to remove unwanted elements from out data like symbols and numbers
from nltk.tokenize import RegexpTokenizer

warnings.filterwarnings('ignore')  # Ignoro i messaggi di warnings

data, X, y = get_dataset()

X = []
sentences = list(data['text'])
for sen in sentences:
    X.append(preprocess_text(sen))

y = data['tag']
y = np.array(list(map(lambda x: 1 if x == "pos" else 0, y)))

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
# Imposto l'opzione ngram_range a (1,2) per considerare anche le parole composte da due termini contigui.
vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), tokenizer=token.tokenize)

X = vect.fit_transform(X)

Sentiment_count = data.groupby('tag').count()
sns.countplot(x='tag', data=data)
plt.xlabel('Sentiments delle recensioni')
plt.ylabel('Numero recensioni')
plt.show()

sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 7))
plt.title('20 termini più frequenti')
plt.xlabel('termini')
plt.ylabel('frequenza')
x_val = [x[0] for x in words_freq[:20]]
y_val = [x[1] for x in words_freq[:20]]

plt.bar(x_val,y_val)
plt.show()
