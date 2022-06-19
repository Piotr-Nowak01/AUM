import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, ShuffleSplit
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
def Statistics(method,file,Testsize=0.3):
    filename = "model_" + method + ".sav"
    if not exists(filename):        #checking if model exists
        print("Nie ma takiego modelu. Próbuję taki stworzyć.")
        test = Training(method,"Tweety.csv")
        if test:
            print("Udało się stworzyć model.")
        else:
            return
    clf = pickle.load(open(filename, 'rb'))
    X_train, X_test, Y_train, Y_test = Get_Data(file,TestSize)
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
def Usage_Combined(file):
    if not exists("model_KNN.sav"):
        print("Tworzenie niezbędnego modelu.")
        Training("KNN","Tweety.csv")
    if not exists("model_MLP.sav"):
        print("Tworzenie niezbędnego modelu.")
        Training("MLP","Tweety.csv")
    if not exists("model_SVC.sav"):
        print("Tworzenie niezbędnego modelu.")
        Training("SVC","Tweety.csv")
    data_to_predict = []
    with open(file, encoding='utf8') as f:
        for line in f:
            data_to_predict.append(line)
    cv = pickle.load(open("CountVectorizer.pickel","rb"))
    X_to_predict = cv.transform(data_to_predict)
    clf1 = pickle.load(open("model_SVC.sav",'rb'))
    clf2 = pickle.load(open("model_MLP.sav",'rb'))
    clf3 = pickle.load(open("model_KNN.sav",'rb'))
    Y1_predicted = clf1.predict(X_to_predict)
    Y2_predicted = clf2.predict(X_to_predict)
    Y3_predicted = clf3.predict(X_to_predict)
    Y_predicted = Y2_predicted*0.4 + Y1_predicted*0.35 + Y3_predicted*0.25
    Y_predicted = [int(val + 0.5) for val in Y_predicted]
    print("Połączone algorytmy.")
    print("Wyniki: " + str(Y_predicted))
def Visualisation(method):
    print("WIP")
    filename = "model_" + method + ".sav"
    if not exists(filename):  # checking if model exists
        print("Nie ma takiego modelu. Próbuję taki stworzyć.")
        test = Training(method, "Tweety.csv")
        if test:
            print("Udało się stworzyć model.")
        else:
            return
    estimator = pickle.load(open(filename, 'rb'))
    title = "Krzywa uczenia modelu "+str(method)
    X, y = load_digits(return_X_y=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    plot_learning_curve(estimator, title, X, y, axes=axes[:], ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,train_sizes=np.linspace(0.1, 1.0, 5)): #this function is taken from scikit-learn library documentaction
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
def Help():
    print("Funkcja \"Training\" pozwala na wytrenowanie modelu na podstawie pliku z danymi. Należy pamiętać, że w każdym wierszu powinien znaleźć się tekst oraz klasyfikacja tekstu. 1 dla tekstów nadających się do publikacji oraz 0, dla tekstów nienadających się do publikacji.")
    print("Funkcja \"Statistics\" pozwala na sprawdzenie współczynników jakości dla danego modelu algorytmu uczenia maszynowego stworzonego na podstawie danych uczących.")
    print("Funkcja \"Usage\" pozwala na wykorzystanie gotowego modelu do predykcji etykiet dla podanych danych. W przypadku, gdyby wcześniej nie stworzono modelu, to program najpierw stworzy model dla danego algorytmu na podstawie wcześniej przygotowanych danych uczących a następnie wykorzysta stworzony model do predykcji podanych danych.")
    print("Funkcja \"Help\" wyświetla to okno z wyjaśnieniem działania poszczególnych funkcji programu.")
    print("Funkcja \"UsageCombined\" działa analogicznie do funkcji \"Usage\", lecz zamiast pojedyńczego algorytmu wyciąga średnią ważoną z predykcji wszystkich 3 modeli. Wagi zostały przydzielone zgodnie z dokładnością algorytmów.")
    print("Funkcja \"Visualisation\" pozwala na wizualizację krzywej uczenia dla danego algorytmu uczenia maszynowego. UWAGA: nie działa dla metody KNN.")
    x = input("Wciśnij dowolny przycisk, aby wrócić do menu głównego.")
    return
def Menu():
    while True:
        print("1 - Training")
        print("2 - Statistics")
        print("3 - Usage")
        print("4 - UsageCombined")
        print("5 - Visualisation")
        print("6 - Help")
        print("7 - End program")
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
            file = input("Podaj nazwę pliku wraz z rozszerzeniem, w którym są dane do klasyfikacji.")
            Usage_Combined(file)
        elif x=='5':
            method = input("Podaj, którą metodę chcesz zwizualizować.")
            Visualisation(method)
        elif x=='6':
            Help()
        elif x=='7':
            return
        else:
            print("Nie ma takiej opcji w programie. Spróbuj ponownie.")
#==================================================
Menu()