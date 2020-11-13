# Machine-Learning-DigitRecognition

In order to run this python code you have to install some python modules: pip install moduleName or pip3 install moduleName

Modules to install:

  1. sklearn 
  2. pandas 
  3. numpy
  
  Testo:
  
  Si consideri il dataset "digit.csv" contenente dati relativi al digit-recognition.
  Il dataset è composto da 42000 righe (+1 di intestazione) contenenti ognuna 785 numeri.
  
  Il primo numero indica il digit rappresentato su una certa riga (etichetta label) e può variare in [0-9].
  I successivi 784 numeri rappresentano la matrice 28*28 del digit scritto a mano.
  - ogni numero indica il valore di grigio del pixel corrispondente e può variare in [0-255]
  
  Si chiede pertanto di :
    - analizzare il dataset con gli strumenti software che sono ritenuti più opportuni
    - classificare con vari classificatori e confronto dei risultati
    - applicare una features selection per evidenziare i pixel più significativi
    - proiettare il dataset sulle 30 features più significative
    - riapplicare la classificazione al dataset ridotto con l'algoritmo che si è mostrato migliore al punto 2
    - confrontare i risultati fra classificazione con tutte le features e le prime 30
    - calcolare il numero minimo KMIN di features da utilizzare affinché l'accuracy della predizione non diminuisca di più di 5 punti percentuali 
        (Es: se con tutte le features ottengo 80%, trovare KMIN tale che l'accuracy il non deve scendere sotto il 75%)
