import pandas as pd  # lettura file csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import w3lib.html as w3
import re
import warnings
import time

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import precision_recall_curve, roc_auc_score
# permette di vettorizzare i dati del dataset di training X
from sklearn.feature_extraction.text import CountVectorizer
# tokenizer to remove unwanted elements from out data like symbols and numbers
from nltk.tokenize import RegexpTokenizer


os.chdir("dataset")

def preprocess_text(sen):
    # Removing html tags
    sentence = w3.remove_tags(sen)

    # Remove punctuations
    sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r'(?:^| )\w(?:$| )', ' ', sentence).strip()

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # test di stemming
    #from nltk.stem.porter import PorterStemmer
    #stem = PorterStemmer()
    #newsen = ''
    #for word in sentence.split():
    #    newsen += stem.stem(word) + ' '
    #sentence = newsen

    return sentence


# restituisce i dati del dataset
def get_dataset (q=""):

    extension = 'txt'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # Combino tutti i file nella cartella
    col_names = ["text", "tag"]
    data = pd.concat([pd.read_csv(f, sep="\t", names=col_names) for f in all_filenames])

    X = []
    sentences = list(data['text'])
    for sen in sentences:
        X.append(preprocess_text(sen))

    if q:
        X.append(preprocess_text(q))

    y = data['tag']
    y = np.array(list(map(lambda x: 1 if x == "pos" else 0, y)))

    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # Imposto l'opzione ngram_range a (1,2) per considerare anche le parole composte da due termini contigui.
    vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), tokenizer=token.tokenize)

    X = vect.fit_transform(X)

    '''
    import keras.preprocessing as kpro
    tokenizer = kpro.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = kpro.sequence.pad_sequences(X, padding='post', maxlen=100)
    '''

    return data, X, y


# Stampa report di classificazione
# Precisione = previsioni corrette / previsioni totali
# Precisione = True Positives / (True Positives + False Positives);
#   una precisione inferiore significa un numero maggiore di falsi positivi
# Recall = True Positives / (True Positives + False Negatives);
#   un basso ricordo significa che il modello contiene molti falsi negativi,
#   ovvero non è stato in grado di identificare correttamente una grande percentuale dei membri della classe.
# Punteggio F1 = Media tra Precisione e Richiamo (i pesi possono essere applicati
#   se una metrica è più importante dell'altra per un caso d'uso specifico)
# Supporto = Numero di osservazioni effettive in quella classe
def modelreport(y_test, clf_pred, clf_score, clf_cv_scores, class_names):
    print('--------------- Classification report ---------------')
    print(metrics.classification_report(y_test, clf_pred, target_names=class_names))
    print("Accuracy score        :", clf_score)
    if clf_cv_scores != '':
        print("Cross-Validation score:", np.mean(clf_cv_scores))



# Visualizza la matrice di confusione per valutare l'accuratezza di una classificazione.
# Le figure mostrano la matrice di confusione con e senza normalizzazione in base alla
# dimensione della classe (numero di elementi in ciascuna classe).
# La normalizzazione può essere utile in caso di squilibrio tra le dimensioni delle classi
# e da un'interpretazione più visiva di quale classe viene classificata erroneamente.
from sklearn.metrics import plot_confusion_matrix
def confmatrix(clf, X_test, y_test, title, class_names):

    titles_options = [(title+"\nWithout normalization", None),
                      (title+"\nNormalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.YlGnBu,
                                     normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()


# Varianza e deviazione standard
def variancedev(cv_scores):
    print('\nVarianza stimata:')
    print('Varianza              : {}'.format(np.var(cv_scores)))
    print('Deviazione standard   : {}'.format(np.std(cv_scores)))
    data = {'varianza': np.var(cv_scores), 'dev. standard': np.std(cv_scores)}
    names = list(data.keys())
    values = list(data.values())
    fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
    axs.bar(names, values)
    plt.show()


# ROC Curve
def roccurve(X, y, classifier):
    # y = label_binarize(y, classes=['neg', 'pos'])
    y = np.array([[0, 1] if k == 1 else [1, 0] for k in y])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    y_score = classifier.fit(X_train, y_train).predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2

    colors = cycle(['tomato', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.grid(color='lightgray', linestyle=':', linewidth=1)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    precision_recall(y_test, y_score, n_classes)


# Precision-Recall Curve
def precision_recall(y_test, y_score, n_classes):
    precision = dict()
    recall = dict()
    colors = cycle(['tomato', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=2, label='class {}'.format(i))

    plt.grid(color='lightgray', linestyle=':', linewidth=1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()
