import time

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import pickle
from os.path import exists
def Get_Data(file,TestSize):
    RandomState = 1
    data=pd.read_csv(file,sep=';')
    X = data['text'].values
    Y = data['grade'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=TestSize, random_state=RandomState)
    cv = CountVectorizer()
    x = cv.fit_transform(X)
    #print(cv.vocabulary_)
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    pickle.dump(cv,open("CountVectorizer.pickel","wb"))
    return X_train, X_test, Y_train, Y_test
def Training(method,file,TestSize=0.3):
    X_train, X_test, Y_train, Y_test = Get_Data(file,TestSize=TestSize)
    if method == "SVC":
        clf = SVC()
    elif method == "KNN":
        decision = input("Czy chcesz zmienić liczbę sąsiadów w algorytmie? \n Wpisz \"-1\" jeśli nie. Jeśli tak, to wpisz nową wartość liczby sąsiadów.")
        if decision != '-1':
            n_neighbors = 5
        else:
            n_neighbors = int(decision)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif method == "MLP":
        decision = input("Czy chcesz zmienić maksymalną liczbę iteracji w algorytmie? \n Wpisz \"-1\" jeśli nie. Jeśli tak, to wpisz nową maksymalną liczbę iteracji.")
        if decision == '-1':
            maxiter = 100
        else:
           maxiter = int(decision)
        clf = MLPClassifier(max_iter=maxiter)
    else:
        print("Nie ma takiej metody.")
        return False
    clf.fit(X_train,Y_train)
    filename = "model_"+method+".sav"
    pickle.dump(clf,open(filename,'wb'))
    return True
def Statistics(method,file):
    filename = "model_" + method + ".sav"
    if not exists(filename):        #checking if model exists
        print("Nie ma takiego modelu. Próbuję taki stworzyć.")
        test = Training(method,"Tweety.csv")
        if test:
            print("Udało się stworzyć model.")
        else:
            return
    clf = pickle.load(open(filename, 'rb'))
    X_train, X_test, Y_train, Y_test = Get_Data(file,0.3)
    start = time.time_ns()
    Y_predicted = clf.predict(X_test)
    end = time.time_ns()
    diff = end-start
    print(diff)
    test = Y_test
    Acc = accuracy_score(test,Y_predicted)
    Prec = precision_score(test,Y_predicted,average=None,zero_division=False)
    F1score = f1_score(test,Y_predicted, average=None)
    Con_matrix = confusion_matrix(test,Y_predicted)
    print("Accuracy: "+str(Acc))
    print("Precision: " + str(Prec))
    print("F1 score: " + str(F1score))
    print("Confusion matrix: "+str(Con_matrix))
    print("Working time in nanoseconds: "+str(diff))
    print("Koniec statystyk.")
def Usage(method,file):
    filename = "model_"+method+".sav"
    if not exists(filename):        #checking if model exists
        print("Nie ma takiego modelu. Próbuję taki stworzyć.")
        test = Training(method,"Tweety.csv")
        if test:
            print("Udało się stworzyć model.")
        else:
            return
    clf = pickle.load(open(filename,'rb'))
    cv = pickle.load(open("CountVectorizer.pickel","rb"))
    data_to_predict = []
    with open(file,encoding='utf8') as f:
        for line in f:
            data_to_predict.append(line)
    X_to_predict = cv.transform(data_to_predict)
    Y_predicted = clf.predict(X_to_predict)
    print("Algorytm: "+str(method))
    print("Wyniki: "+str(Y_predicted))
def Visualisation(method):
    print("WORK IN PROGRESS")
def Help():
    print("Funkcja \"Training\" pozwala na wytrenowanie modelu na podstawie pliku z danymi. Należy pamiętać, że w każdym wierszu powinien znaleźć się tekst oraz klasyfikacja tekstu. 1 dla tekstów nadających się do publikacji oraz 0, dla tekstów nienadających się do publikacji.")
    print("Funkcja \"Statistics\" pozwala na sprawdzenie współczynników jakości dla danego modelu algorytmu uczenia maszynowego stworzonego na podstawie danych uczących.")
    print("Funkcja \"Usage\" pozwala na wykorzystanie gotowego modelu do predykcji etykiet dla podanych danych. W przypadku, gdyby wcześniej nie stworzono modelu, to program najpierw stworzy model dla danego algorytmu na podstawie wcześniej przygotowanych danych uczących a następnie wykorzysta stworzony model do predykcji podanych danych.")
    print("Funkcja \"Help\" wyświetla to okno z wyjaśnieniem działania poszczególnych funkcji programu.")
    print("Funkcja \"Visualisation\" pozwala na wizualizację krzywych uczenia dla poszczególnych metod uczenia maszynowego.")
    x = input("Wciśnij dowolny przycisk, aby wrócić do menu głównego.")
    return
def Menu():
    while True:
        print("1 - Training")
        print("2 - Statistics")
        print("3 - Usage")
        print("4 - Visualisation")
        print("5 - Help")
        print("6 - End program")
        print("Wpisz liczbę, aby uruchomić daną funkcję.")
        x=input()
        if x=='1':
            method = input("Podaj algorytm, jaki chcesz użyć. \n Dostępne: SVC, MLP, KNN. \n")
            file = input("Podaj nazwę pliku z rozszerzeniem csv, w którym znajdują się dane uczące.")
            Training(method,file)
        elif x=='2':
            method = input("Podaj algorytm, jaki chcesz użyć. \n Dostępne: SVC, MLP, KNN. \n")
            file = input("Podaj nazwe pliku z rozszerzeniem csv, w którym znajdują się dane testowe.")
            Statistics(method,file)
        elif x=='3':
            method = input("Podaj algorytm, jaki chcesz użyć. \n Dostępne: SVC, MLP, KNN. \n")
            file = input("Podaj nazwę pliku wraz z rozszerzeniem, w którym są dane do klasyfikacji.")
            Usage(method,file)
        elif x=='4':
            method = input("Podaj algorytm, jaki chcesz zwizualizować. \n Dostępne: SVC, MLP, KNN. \n")
            Visualisation(method)
        elif x=='5':
            Help()
        elif x=='6':
            return
        else:
            print("Nie ma takiej opcji w programie. Spróbuj ponownie.")
#==================================================
Menu()