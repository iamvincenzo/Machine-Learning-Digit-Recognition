# moduli usati per importare classificatori supervisionati

from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# modulo usato per importare classificatori non supervisionati

from sklearn.cluster import KMeans

# moduli usato per manipolare il dataset

import pandas as pd
import numpy as np

# moduli usati per la valutazione del classificatore

# modulo usato per splittare il dataset in train-set / test-set in maniera bilanciata

from sklearn.model_selection import train_test_split

# modulo usato per creare la matrice di confusione: una matrice che mi indica le classi più sbagliate dall'algoritmo --> se una classe è molto preponderante rispetto all'altra vuol dire che il dataset non è bilanciato bene in quanto l'algoritmo apprende troppo da quella classe e meno dalle altre

from sklearn.metrics import confusion_matrix

# fine moduli usati per la valutazione del classificatore


# moduli usati per selezionare le features più importanti

from sklearn.feature_selection import SelectKBest, f_classif

# from sklearn.model_selection import train_test_split --> già incluso prima

# modulo usato per misurare i tempi di addestramento

import time

# modulo usato per disabilitare i warnings in python

import warnings

warnings.filterwarnings('ignore')  # disabilito eventuali warnings in python

# 1. Analizzare il dataset con gli strumenti software che sono ritenuti più opportuni

data = pd.read_csv("digit.csv", sep=",")  # carico il dataset

print("\nDATA_HEAD: \n", data.head(), "\n")

print(data.keys())  # siamo in grado di eseguire questo comando perchè automaticamente python isola questa riga (la prima) dall'intero dataset --> tale riga contiene i nomi delle features e il nome label che indica le labels delle features

label = set(data['label'])  # prelevo solo le label (numeri decimali rappresentati nel dataset) senza ripetizioni grazie al comando set

print("LABELS: \n", label, "\n")

X = data.iloc[:, 1:]  # prelevo solamente le features dal dataset. Cioè tutti i dati --> tutte le righe e tutte le colonne a partire dalla seconda (la prima contiene le labels)

print("FEATURES: \n", X, "\n")

y = data['label']  # assegno ad y le labels

print("LABELS: \n", y, "\n")

# faccio separare a python in maniera bilanciata i dati tra test-set e training-set, in questo caso poichè il 
# dataset contiene molti elementi non occorre usare il cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # 0.33 indica che il test-set contierrà il 33% dei dati del dataset --> di conseguenza il training set il 67%

# 2. classificare con vari classificatori

# Classificatore con apprendimento supervisionato

clfRF = tree.DecisionTreeClassifier()  # creo un'istanza del classificatore Random Forest
clfSVM = SVC(gamma='auto', probability=True)  # creo un'istanza del classificatore basato su Supported Vector Machine
clfNB = MultinomialNB()  # creo un'istanza del classificatore naive_bayes basasto sulla formula di teoria dell probabilità di Bayes

# Classificatore con apprendimento non supervisionato

kmeans = KMeans(n_clusters=10, random_state=0)  # K = 10 perchè ho numeri da 0 a 9

# Addestramento classificatore con apprendimento supervisionato

clfRF = clfRF.fit(X_train, y_train)  # faccio fittare (imparare dai dati) il classificatore Random Forset

'''
clfNB = clfNB.fit(X_train, y_train) # faccio fittare (imparare dai dati il classificatore Naive Bayes)
clfSVM.fit(X_train, y_train) # è molto lenta --> non va bene!!!

# Addestramento classficatore con apprendimento non supervisionato

kmeans.fit(X_train) # algoritmo clusterizza X in 10 gruppi e impara
'''

# 2. e confronto dei risultati --> valutazione classificatore: si valuta la bontà e la precisione dell'algoritmo con i dati a disposizione.

print("X_train len: " + str(len(X_train)), "y_train len: " + str(len(y_train)))  # stampo la lunghezza di X_train ed y_train

print("X_test len: " + str(len(X_test)), "y_test len: " + str(len(y_test)))  # stampo la lunghezza di X_test ed y_test

print("\nRandom Forest: ")  # vedo quante volte il classificatore ci prende --> cioè l'accuracy del classificatore rispettivamente nel test-set e training-set

print("TRAINING-SET: ", clfRF.score(X_train, y_train), "error: ", (1 - clfRF.score(X_train, y_train)) * 100, "%")
print("TEST-SET: ", clfRF.score(X_test, y_test), "error: ", (1 - clfRF.score(X_test, y_test)) * 100, "%")

prediction_accuracy = clfRF.score(X_test, y_test)  # salvo tale valore perchè servirà in seguito per confrontarlo con quello calcolato con il dataset con meno features

print("Prediction accuracy: ", prediction_accuracy)

# dove sono stati commessi gli errori

print("\nErrors in training-set Random Forest: \n")

predictions = clfRF.predict(X_train)  # crea un array contenente le su predictions sui dati

for elem, prediction, label in zip(X_train, predictions, y_train):  # confronto le predictions dell'algoritmo con i risultati corretti
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

print("\nErrors in test-set Random Forest: \n")

predictions = clfRF.predict(X_test)

for elem, prediction, label in zip(X_test, predictions, y_test):
    if prediction != label:
        print(elem, " has been classified as ", prediction, " and should be ", label)

# matrice di confusione --> mostra quanti errori in ogni classe. Mostra quanti elementi di quella classe sono stati inseriti nella classe sbagliata

cmRF_training = confusion_matrix(y_train, clfRF.predict(X_train))

print("\nCM per Training-set Random Forest: \n", cmRF_training, "\n")

cmRF_test = confusion_matrix(y_test, clfRF.predict(X_test))

print("\nCM per Test-set Random Forest: \n", cmRF_test, "\n")

# 3. applicare una features selection per evidenziare i pixel più significativi

feature_importances_ = clfRF.feature_importances_

print("Importanza assegnata alle features dall'algoritmo (indica quanto gli sono servite durante il suo allenamento): \n", feature_importances_, "\n")

# 4. proiettare il dataset sulle 30 features più significative

select = SelectKBest(f_classif, k=30)
select.fit(X, y)

mask = select.get_support()  # ottengo un array che è composto da booleani. 'True' se la feature è importante 'False' se non è importante

np_mask = np.array(mask)  # trasformo mask in un array numpy per poter eseguire operazioni su tale array --> in particolare l'estrazione delle fetures più significative

np_columns = np.array(data.columns[1:])  # prelevo tutte le colonne del dataset, la prima è "label" e non va considerata

most_significative_features = np_columns[np_mask]  # selezioniamo le features più significative

most_significative_features_importances = clfRF.feature_importances_[np_mask]  # seleziono i rispettivi valori di importanza

print("Features più significative: \n", most_significative_features, "\n")
print("Importanza delle features più significative: \n", most_significative_features_importances, "\n")

importance_per_feature = zip(most_significative_features, most_significative_features_importances)  # accoppiamo i feature con le relative importanze e le stampiamo

'''
for i in importance_per_feature:
    print(i)
'''

'''
# esempio per capire cosa succede

f = [True, False, False, True]

np_f = np.array(f)

np_columns = np.array([10, 20, 14, 30])

np_columns_importances = np_columns[np_f] # tale prodotto serve per realizzare l'estrazione dei valori contenuti in np_columns che hanno il corrispondente valore true in np_f

print(np_columns_importances)
'''

# 5. riapplicare la classificazione al dataset ridotto con l'algoritmo che si è mostrato migliore al punto 2

X_reduced = data[most_significative_features]  # selezionare le righe passando come parametro una lista di colonne

# le labels del dataset ridotto rimangono invece le stesse dell'origiale --> poichè riduciamo il numero delle colonne non delle righe
y_reduced = y

print(X_reduced.head())

"""
CLASSIFICAZIONE CON LE SOLE 30 FEATURES PIÙ SIGNIFICATIVE --> stessi procedimenti precedenti ma con il dataset ridotto
"""

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.33, random_state=42)

clfRF_new = tree.DecisionTreeClassifier()

clfRF_new.fit(X_train_reduced, y_train_reduced)  # training del classificatore con il dataset ridotto

new_predictions = clfRF_new.predict(X_test_reduced)  # ottengo le predizioni con il dataset ridotto

prediction_accuracy_reduced = clfRF_new.score(X_test_reduced, y_test_reduced)  # ottengo l'accuracy del classificatore con li dataset ridotto

# 6.  confrontare i risultati fra classificazione con tutte le features e le prime 30

print("Accuracy con tutte le features: ", prediction_accuracy, "\n")

print("Accuracy con le features più significative (dataset ridotto): ", prediction_accuracy_reduced, "\n")

"""
Comparazione fra diversi classificatori

addestriamo anche classificatori con algoritmi diversi:
- SVM (Support vector machine)
- Random Forest
- NN (Neural Network)
- NB (Naive Bayes)

utilizziamo per tutti il medesimo train e test set ridotto

confrontiamo le accuracy di previsione e i tempi di calcolo
"""

for clf in [tree.DecisionTreeClassifier(),  # struttura che contiene i classificatori
            MultinomialNB(),
            svm.LinearSVC(),
            MLPClassifier(random_state=1, max_iter=300)
            ]:
    start = time.time()
    clf.fit(X_train_reduced, y_train_reduced)
    accuracy = clf.score(X_test_reduced, y_test_reduced)
    print("Classificatore: ", clf, "- Accuracy: ", accuracy, "- Tempo richiesto per l'addestramento: ", round(time.time() - start, 2), "seconds")

# 7.  calcolare il numero minimo KMIN di features da utilizzare affinché l'accuracy della predizione non diminuisca di più di 5 punti percentuali
# (Es: se con tutte le features ottengo 80%, trovare KMIN tale che l'accuracy il non deve scendere sotto il 75%)

"""
Ricerca del numero minimo di features necessarie per non scendere
sotto l'80% (5% in meno di quanto ottenuto con l'albero di decisione e tutte le features)
"""

warnings.filterwarnings('ignore')

KBEST = 30  # uso una variabile per indicare quante features voglio selezionare
# incrementiamo di 5 per volta finché l'accuratezza non sale sopra all' 80%

while 1:
    select = SelectKBest(f_classif, k=KBEST)  # ogni volta il valore di KBEST cambia
    select.fit(X, y)
    mask = select.get_support()
    np_mask = np.array(mask)
    np_columns = np.array(data.columns[1:])  # la prima è "label" e non va considerata...
    most_significative_features = np_columns[np_mask]

    # dataset ridotto
    X_reduced = data[most_significative_features]
    y_reduced = y

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.33, random_state=42)
    clfRF_new = tree.DecisionTreeClassifier()

    clfRF_new.fit(X_train_reduced, y_train_reduced)  # training del classificatore

    prediction_accuracy_reduced = clfRF_new.score(X_test_reduced, y_test_reduced)  # accuracy con K features

    print("Accuracy con {} features: {} %".format(KBEST, round(prediction_accuracy_reduced, 2)))

    if prediction_accuracy_reduced > 0.8:  # tolleriamo un margine perdita del 5% di accuracy con meno features
        break
    else:
        KBEST += 5
