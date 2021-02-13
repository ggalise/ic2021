from funzioni import *
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')  # Ignoro i messaggi di warnings

data, X, y = get_dataset()

# Il test set è composto dal 30% degli esempi. Servirà per provare l'affidabilità del modello.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Calcola l'errore per K valori tra 1 e 20
error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Grafico che mostra l'errore medio nelle predizioni a seguito di una variazione del valore K (numero vicini)
plt.plot(range(1, 20), error, color='tomato', linestyle='solid', marker='.', markerfacecolor='firebrick', markersize=10)
plt.title('Tasso di errore valore K')
plt.grid(color='lightgray', linestyle=':', linewidth=1)
plt.xlabel('Valore K')
plt.ylabel('Media errore')
plt.show()

start = time.time()

# Instanzio l'algoritmo
# imposto a 6 i nodi k
clf = KNeighborsClassifier(n_neighbors=6)

class_names = ['0 (neg)', '1 (pos)']
modelname = 'KNeighborsClassifier'

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