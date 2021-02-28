import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedKFold


def arrumar_nome(sv, X,y):
    __svm = sv
    ind_k_fold = rkf.split(X)# tem que definir sempre que for usar
    acuracias= []
    for train_index, test_index in ind_k_fold:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        __svm.fit(X_train,y_train)
        y_mm = __svm.predict(X_test)
        scores_svm = classification_report(y_test,y_mm,output_dict=True) 
        acuracias.append(scores_svm['accuracy'])

    return acuracias


id_col = ['RI', 'Na', 'Mg', 'Al', 'Si','K', 'Ca', 'Ba', 'Fe', 'Type_of_glass']
df = pd.read_csv('base/glass.csv',  encoding='UTF-8',sep=',', names=id_col, index_col=0)

#X = pd.DataFrame(pre.normalize(df.iloc[:, 0:9], norm='l2'), columns=id_col[1:10])
X = minmax_scale(df.iloc[:, 0:9].values)
y = df.iloc[:, -1].values

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=2652124)

max_C = X.shape[0]


acuracia_Cs =[[x, arrumar_nome(svm.SVC(C=4, kernel='poly', degree=83),X,y)] for x in range(1,max_C)]

print(np.array(acuracia_Cs))
'''
best_C = []
for i in acuracia_Cs:
    temp = [[i[0], np.array(i[1]).mean()]]
    if len(best_C)>1 and temp[0][1]==best_C[0][1] :
        temp.extend([x for x in best_C])
        best_C = temp
        continue
    best_C.append(temp[0])

#sorted(best_C,key=lambda x: x[1])
print(best_C[-10:])'''