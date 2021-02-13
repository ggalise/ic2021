from funzioni import *
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
warnings.filterwarnings('ignore')

data, X, y = get_dataset()

# Il test set è composto dal 30% degli esempi. Servirà per provare l'affidabilità del modello.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

start = time.time()

# Instanzio l'algoritmo
clf = Perceptron(max_iter=1000)

class_names = ['0 (neg)', '1 (pos)']
modelname = 'Perceptron'

# Adatto il modello ai dati
clf.fit(X_train, y_train)

# Effettuo predizioni sul set di prova
clf_pred = clf.predict(X_test)

# Ottengo le metriche sulle prestazioni
clf_score = metrics.accuracy_score(y_test, clf_pred) * 100

# Cross-Validation score
# cv = Determines the cross-validation splitting strategy
clf_cv_scores = cross_val_score(clf, X, y, cv=15) * 100

end = time.time()
print('Execution time: ', end - start, "seconds\n")

# Stampa report di classificazione
modelreport (y_test, clf_pred, clf_score, clf_cv_scores, class_names)

# Matrice confusione
confmatrix(clf, X_test, y_test, "Confusion Matrix - "+modelname, class_names)

# Varianza e deviazione standard
variancedev(clf_cv_scores)

# Questa strategia consiste nell'adattare un classificatore per classe.
# Per ogni classificatore, la classe viene confrontata con tutte le altre classi.
classifier = OneVsRestClassifier(clf)

# ROC Curve (include Precision-Recall Curve)
roccurve(X, y, classifier)