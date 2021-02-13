from funzioni import *
from sklearn.naive_bayes import MultinomialNB

q = input("Enter your text: ")
#q = 'I found this place by accident and I could not be happier.'
#q = 'seems like a good quick place to grab a bite of some familiar pub food, but do yourself a favor and look elsewhere.'

data, X, y = get_dataset(q)

array_length = X.shape[0]
q = X[array_length - 1]
X = X[:-1]

# Instanzio l'algoritmo
clf = MultinomialNB()
clf.fit(X, y)
q_pred = clf.predict(q)

if q_pred == 0:
    print('\nNegative Sentiment')
else:
    print('\nPositive Sentiment')
