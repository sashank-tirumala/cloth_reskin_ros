import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import sklearn
import pickle
classification_folder = "/home/sashank/catkin_ws/src/tactilecloth/classification_data"
folders = ["0cloth_7feb","1cloth_7feb", "2cloth_7feb"]
num_data = 10
train_data = []
test_data_set = [11,12,13,14,15]
for fn in folders:
    for i in range(num_data):
        name = classification_folder+"/"+fn+"/"+str(i+1)+"/"+str(i+1)+"_reskin_data.csv"
        marker_name = classification_folder+"/"+fn+"/"+str(i+1)+"/"+str(i+1)+"_markers.csv"
        nparr = np.loadtxt(name, delimiter=",")
        markerarr = np.loadtxt(marker_name, delimiter=",")
        train_data.append(nparr)
train_data = np.vstack(train_data)
x_train = train_data[:,:-2]
y_train = train_data[:,-2]
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
test_data_set = np.arange(num_data+1, 16)
test_data = []
for fn in folders:
    for i in test_data_set:
        name = classification_folder+"/"+fn+"/"+str(i)+"/"+str(i)+"_reskin_data.csv"
        marker_name = classification_folder+"/"+fn+"/"+str(i)+"/"+str(i)+"_markers.csv"
        nparr = np.loadtxt(name, delimiter=",")
#         markerarr = np.loadtxt(marker_name, delimiter=",")
#         test_data.append(nparr[int(markerarr[0]*17-20):int(markerarr[0]*17+20),:])
        test_data.append(nparr)
test_data = np.vstack(test_data)
x_test = test_data[:,:-2]
y_test = test_data[:,-2]
x_test = scaler.transform(x_test)
clf = neighbors.KNeighborsClassifier(10, weights="distance")
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x_test)
best_cf = confusion_matrix(y_test, y_pred)
print(best_cf)
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)
with open("/home/sashank/catkin_ws/src/tactilecloth/scripts/clf.pickle", 'wb') as handle:
    pickle.dump(clf, handle, protocol=2)