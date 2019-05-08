import pandas
import numpy
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

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
from Machine_learning_begin import Explained_Variance_fig

names=['Age','Sex','Height',' Weight','QRS duration','P-R interval','Q-T interval','T interval','P interval','QRS','T','P','QRST','J','Heart rate',
'Q wave','R wave','S wave',"R' wave","S' wave",'Number of intrinsic deflections','Existence of ragged R wave','Existence of diphasic derivation of R wave',
'Existence of ragged P wave','Existence of diphasic derivation of P wave','Existence of ragged T wave','Existence of diphasic derivation of T wave',
'Q wave DII','R wave DII','S wave DII',"R' wave DII","S' wave DII",'Number of intrinsic deflections DII','Existence of ragged R wave DII','Existence of diphasic derivation of R wave DII',
'Existence of ragged P wave DII','Existence of diphasic derivation of P wave DII','Existence of ragged T wave DII','Existence of diphasic derivation of T wave DII',
'Q wave DIII','R wave DIII','S wave DIII',"R' wave DIII","S' wave DIII",'Number of intrinsic deflections DIII','Existence of ragged R wave DIII','Existence of diphasic derivation of R wave DIII',
'Existence of ragged P wave DIII','Existence of diphasic derivation of P wave DIII','Existence of ragged T wave DIII','Existence of diphasic derivation of T wave DIII',
'Q wave AVR','R wave AVR','S wave AVR',"R' wave AVR","S' wave AVR",'Number of intrinsic deflections AVR','Existence of ragged R wave AVR','Existence of diphasic derivation of R wave AVR',
'Existence of ragged P wave AVR','Existence of diphasic derivation of P wave AVR','Existence of ragged T wave AVR','Existence of diphasic derivation of T wave AVR',
'Q wave AVL','R wave AVL','S wave AVL',"R' wave AVL","S' wave AVL",'Number of intrinsic deflections AVL','Existence of ragged R wave AVL','Existence of diphasic derivation of R wave AVL',
'Existence of ragged P wave AVL','Existence of diphasic derivation of P wave AVL','Existence of ragged T wave AVL','Existence of diphasic derivation of T wave AVL',
'Q wave AVF','R wave AVF','S wave AVF',"R' wave AVF","S' wave AVF",'Number of intrinsic deflections AVF','Existence of ragged R wave AVF','Existence of diphasic derivation of R wave AVF',
'Existence of ragged P wave AVF','Existence of diphasic derivation of P wave AVF','Existence of ragged T wave AVF','Existence of diphasic derivation of T wave AVF',
'Q wave V1','R wave V1','S wave V1',"R' wave V1","S' wave V1",'Number of intrinsic deflections V1','Existence of ragged R wave V1','Existence of diphasic derivation of R wave V1',
'Existence of ragged P wave V1','Existence of diphasic derivation of P wave V1','Existence of ragged T wave V1','Existence of diphasic derivation of T wave V1',
'Q wave V2','R wave V2','S wave V2',"R' wave V2","S' wave V2",'Number of intrinsic deflections V2','Existence of ragged R wave V2','Existence of diphasic derivation of R wave V2',
'Existence of ragged P wave V2','Existence of diphasic derivation of P wave V2','Existence of ragged T wave V2','Existence of diphasic derivation of T wave V2',
'Q wave V3','R wave V3','S wave V3',"R' wave V3","S' wave V3",'Number of intrinsic deflections V3','Existence of ragged R wave V3','Existence of diphasic derivation of R wave V3',
'Existence of ragged P wave V3','Existence of diphasic derivation of P wave V3','Existence of ragged T wave V3','Existence of diphasic derivation of T wave V3',
'Q wave V4','R wave V4','S wave V4',"R' wave V4","S' wave V4",'Number of intrinsic deflections V4','Existence of ragged R wave V4','Existence of diphasic derivation of R wave V4',
'Existence of ragged P wave V4','Existence of diphasic derivation of P wave V4','Existence of ragged T wave V4','Existence of diphasic derivation of T wave V4',
'Q wave V5','R wave V5','S wave V5',"R' wave V5","S' wave V5",'Number of intrinsic deflections V5','Existence of ragged R wave V5','Existence of diphasic derivation of R wave V5',
'Existence of ragged P wave V5','Existence of diphasic derivation of P wave V5','Existence of ragged T wave V5','Existence of diphasic derivation of T wave V5',
'Q wave V6','R wave V6','S wave V6',"R' wave V6","S' wave V6",'Number of intrinsic deflections V6','Existence of ragged R wave V6','Existence of diphasic derivation of R wave V6',
'Existence of ragged P wave V6','Existence of diphasic derivation of P wave V6','Existence of ragged T wave V6','Existence of diphasic derivation of T wave V6',
'JJ wave AMPL','Q wave AMPL','R wave AMPL','S wave AMPL',"R' wave AMPL","S' wave AMPL",'P wave AMPL','T wave AMPL','QRSA AMPL','QRSTA AMPL',
'JJ wave AMPL DII','Q wave AMPL DII','R wave AMPL DII','S wave AMPL DII',"R' wave AMPL DII","S' wave AMPL DII",'P wave AMPL DII','T wave AMPL DII','QRSA AMPL DII','QRSTA AMPL DII',
'JJ wave AMPL DIII','Q wave AMPL DIII','R wave AMPL DIII','S wave AMPL DIII',"R' wave AMPL DIII","S' wave AMPL DIII",'P wave AMPL DIII','T wave AMPL DIII','QRSA AMPL DIII','QRSTA AMPL DIII',
'JJ wave AMPL AVR','Q wave AMPL AVR','R wave AMPL AVR','S wave AMPL AVR',"R' wave AMPL AVR","S' wave AMPL AVR",'P wave AMPL AVR','T wave AMPL AVR','QRSA AMPL AVR','QRSTA AMPL AVR',
'JJ wave AMPL AVL','Q wave AMPL AVL','R wave AMPL AVL','S wave AMPL AVL',"R' wave AMPL AVL","S' wave AMPL AVL",'P wave AMPL AVL','T wave AMPL AVL','QRSA AMPL AVL','QRSTA AMPL AVL',
'JJ wave AMPL AVF','Q wave AMPL AVF','R wave AMPL AVF','S wave AMPL AVF',"R' wave AMPL AVF","S' wave AMPL AVF",'P wave AMPL AVF','T wave AMPL AVF','QRSA AMPL AVF','QRSTA AMPL AVF',
'JJ wave AMPL V1','Q wave AMPL V1','R wave AMPL V1','S wave AMPL V1',"R' wave AMPL V1","S' wave AMPL V1",'P wave AMPL V1','T wave AMPL V1','QRSA AMPL V1','QRSTA AMPL V1',
'JJ wave AMPL V2','Q wave AMPL V2','R wave AMPL V2','S wave AMPL V2',"R' wave AMPL V2","S' wave AMPL V2",'P wave AMPL V2','T wave AMPL V2','QRSA AMPL V2','QRSTA AMPL V2',
'JJ wave AMPL V3','Q wave AMPL V3','R wave AMPL V3','S wave AMPL V3',"R' wave AMPL V3","S' wave AMPL V3",'P wave AMPL V3','T wave AMPL V3','QRSA AMPL V3','QRSTA AMPL V3',
'JJ wave AMPL V4','Q wave AMPL V4','R wave AMPL V4','S wave AMPL V4',"R' wave AMPL V4","S' wave AMPL V4",'P wave AMPL V4','T wave AMPL V4','QRSA AMPL V4','QRSTA AMPL V4',
'JJ wave AMPL V5','Q wave AMPL V5','R wave AMPL V5','S wave AMPL V5',"R' wave AMPL V5","S' wave AMPL V5",'P wave AMPL V5','T wave AMPL V5','QRSA AMPL V5','QRSTA AMPL V5',
'JJ wave AMPL V6','Q wave AMPL V6','R wave AMPL V6','S wave AMPL V6',"R' wave AMPL V6","S' wave AMPL V6",'P wave AMPL V6','T wave AMPL V6','QRSA AMPL V6','QRSTA AMPL V6','class']

class_code={'Normal':1,'Ischemic changes (Coronary Artery Disease)':2,'Old Anterior Myocardial Infarction':3,'Old Inferior Myocardial Infarction':4,'Sinus tachycardy':5,'Sinus bradycardy':6,
'Ventricular Premature Contraction (PVC)':7,
'Supraventricular Premature Contraction':8,
'Left bundle branch block':9,
'Right bundle branch block':10,
'1. degree AtrioVentricular block':11,
'2. degree AV block':12,
'3. degree AV block':13,
'Left ventricule hypertrophy':14,
'Atrial Fibrillation or Flutter':15,
'Others':16	}

features=names[0:(len(names)-1)]
classes_l=['class']

def myplot(score,coeff,y,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


def PCA_components(dataset, features, classes_l, calculate_y=0):
    data = dataset[features]
    classes = numpy.array(dataset[classes_l])
    y = numpy.zeros(len(classes))
    a = 0
    if calculate_y == 1:
        for i in classes:
            if i[0] == 'Iris-setosa':
                y[a] = 0
            if i[0] == 'Iris-versicolor':
                y[a] = 1
            if i[0] == 'Iris-virginica':
                y[a] = 2
            a += 1
    # scale data (normalization)
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    # PCA

    pca = PCA(0.9) # pca è una classe con attributi e metodi
    new_scaled_data = pca.fit_transform(scaled_data)

    return new_scaled_data, pca, y

# names=['Age','Sex','Height',' Weight','QRS duration','P-R interval','Q-T interval','T interval','P interval','QRS','T','P','QRST','J','Heart rate']








if __name__ == "__main__":

    ### COSE DA FARE: 1)PCA DEL DATASET COSI DA RIDURRE DIMENSIONE, 2)
    pandas.set_option('display.max_columns', None)

    # Loading data
    filename = 'arrhythmia.data'
    dataset = pandas.read_csv(filename,names=names)  # conoscendo i nomi delle colonne è utile separarle con i loro nomi inclusi

    # Assessing missing data values
    # Revise this website after reading the article where these data are used (vedere come fanno fronte a tale problema)
    # https://machinelearningmastery.com/handle-missing-data-python/

    dataset = dataset.replace('?', numpy.NaN) #replace '?' with Nan
    dataset.dropna(inplace=True) # drops all the columns with Nan
    # dataset.fillna(0, inplace=True) # fills Nan with the mean of that column (NON FUNZIONA SBATTONE.. CAPIRE SE METTENDO ZERI LA COSA è ATTENDIBILE)
    # print(dataset)

    for i in ['T','P','QRST','J','Heart rate']:
        dataset[i] = dataset[i].astype(int)

    ### PCA ###

    new_scaled_data, pca, y=PCA_components(dataset,features,classes_l)
    Explained_Variance_fig(pca.explained_variance_ratio_)

    # myplot(new_scaled_data[:, 0:2], numpy.transpose(pca.components_[0:2, :]), y)
    # plt.show()



















    ## count the number of NaN values in each column

    # print(dataset.describe())
    # print(dataset.isnull().sum())
    # print(dataset.shape)



    # # shape
    # print(dataset.shape) # dimensione del dataset
    # # head
    # print(dataset.head(20),) # mostra le prime 20 righe
    # # descriptions
    # print(dataset.describe()) # ritorna una serie di parametri statistici utili
    # # class distribution
    # print(dataset.groupby('Age').size())  # ragroppa nelle classi trovate nella colonna "class" e ne dà la numerosità
    #
    #
    # #box and whisker plots
    #
    # dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # plt.show()
    #
    # # histograms
    # dataset.hist()
    # plt.show()

    # # scatter plot matrix
    # scatter_matrix(dataset)
    # plt.show()


   #  # Split-out validation dataset 80% training 20% validation
   #  # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
   #
   #  array = dataset.values #to_numpy()
   #  X = array[:, 0:(len(names)-1)]
   #  Y = array[:, (len(names)-1)]
   #
   #  Y = Y.astype('float') # if there is an error try this
   #
   #  validation_size = 0.20
   #  seed = 7
   #  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,random_state=seed)
   #
   #  # Test options and evaluation metric
   #  seed = 7
   #  scoring = 'accuracy'
   #
   #  # Spot Check Algorithms
   #  models = []
   #  # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
   #  # models.append(('LDA', LinearDiscriminantAnalysis()))
   #  models.append(('KNN', KNeighborsClassifier()))
   #  models.append(('CART', DecisionTreeClassifier()))
   #  # models.append(('NB', GaussianNB()))
   #  # models.append(('SVM', SVC(gamma='auto')))
   #
   #
   #  # evaluate each model in turn
   #
   #  results = []
   #  names = []
   #  for name, model in models:
   #      kfold = model_selection.KFold(n_splits=100, random_state=seed)
   #      cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
   #      results.append(cv_results)
   #      names.append(name)
   #      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   #      print(msg)
   #
   #  # # Make predictions on validation dataset
   #
   #  knn = KNeighborsClassifier()
   #  knn.fit(X_train, Y_train)
   #  predictions = knn.predict(X_validation)
   # # print(accuracy_score(Y_validation, predictions))
   #  # print(confusion_matrix(Y_validation, predictions))
   #  print(classification_report(Y_validation, predictions))