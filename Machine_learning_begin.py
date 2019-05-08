# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Packages

import numpy
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def myplot(score,coeff,y,features,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, features[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

def PCA_components(dataset,features,classes_l,calculate_y=0):

        data = dataset[features]
        classes = numpy.array(dataset[classes_l])
        y = numpy.zeros(len(classes))
        a = 0
        if calculate_y==1:
            for i in classes:
                if i[0] == 'Iris-setosa':
                    y[a] = 0
                if i[0] == 'Iris-versicolor':
                    y[a] = 1
                if i[0] == 'Iris-virginica':
                    y[a] = 2
                a += 1
        # scale data
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        # PCA
        pca = PCA(2) # tiene solo le componenti che spiegano il 95% della varianza
        new_scaled_data = pca.fit_transform(data) # proietta i dati lungo le componenti

        return new_scaled_data,pca,y

def Explained_Variance_fig(explained_variance):
    fig, ax = plt.subplots()
    plt.title('Explained Variance')
    plt.xlabel('Components')
    plt.ylabel('% Explained Variance')
    ax.bar(numpy.linspace(1, len(explained_variance), num=len(explained_variance)), explained_variance, width=1,tick_label=numpy.linspace(1, len(explained_variance), num=len(explained_variance)), align='center')
    plt.show()







if __name__ == "__main__":
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    features=['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    classes_l=['class']
    dataset = pandas.read_csv(url, names=names) # conoscendo i nomi delle colonne è utile separarle con i loro nomi inclusi
    # features of the dataset
    new_scaled_data,pca,y=PCA_components(dataset,features,classes_l,calculate_y=1) # new_scaled_data SONO I DATI ORIGINALI PROIETTATI LUNGO LE COMPONENTI PCA

    # Explained Variance Figure
    explained_variance=pca.explained_variance_ratio_
    Explained_Variance_fig(explained_variance)

    # Dati proiettati  e componenti

    myplot(new_scaled_data[:, 0:2], numpy.transpose(pca.components_[0:2, :]),y,features)
    plt.show()

    # IMPLEMENTARE QUESTE DUE COSE DELLA PCA IN MANIERA GENERALE MAGARI O AVERE UN FILE CHE TI SPIEGA QUESTA COSA








    # # shape
    # print(dataset.shape) # dimensione del dataset
    # # head
    # print(dataset.head(20),) # mostra le prime 20 righe
    # # descriptions
    # print(dataset.describe()) # ritorna una serie di parametri statistici utili
    # # class distribution
    # print(dataset.groupby('class').size())  # ragroppa nelle classi trovate nella colonna "class" e ne dà la numerosità
    #
    #
    # # # box and whisker plots
    # #
    # # dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # # plt.show()
    # #
    # # # histograms
    # # dataset.hist()
    # # plt.show()
    # #
    # # # scatter plot matrix
    # # scatter_matrix(dataset)
    # # plt.show()
    #
    # # Split-out validation dataset 80% training 20% validation
    #
    # # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
    #
    # array = dataset.values #to_numpy()
    # X = array[:, 0:4]
    # Y = array[:, 4]
    # validation_size = 0.20
    # seed = 7
    # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,random_state=seed)
    #
    # # Test options and evaluation metric
    # seed = 7
    # scoring = 'accuracy'
    #
    # # Spot Check Algorithms
    # models = []
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto')))
    #
    #
    # # evaluate each model in turn
    #
    # results = []
    # names = []
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    #
    # # Make predictions on validation dataset
    # knn = KNeighborsClassifier()
    # knn.fit(X_train, Y_train)
    # predictions = knn.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))