import numpy as np
import sklearn.cluster as clust
import sklearn.preprocessing as prepro
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

f = open("/Users/blindfish/bank/dataset.tsv")

data = []

for i in f.read().split("\n"):
    data.append(i.split("\t"))

data = data[:-1]
cdata = []
for i in data:
    cdata.append(i[0:8])

def category2vec(various):
    result = np.array([])
    various_set = list(set(various))
    for i in various:
        rows = np.zeros(len(various_set))
        for j in range(len(various_set)):
            if i == various_set[j]:
                rows[j]=1
        result = np.append(result,rows)
    return result.reshape(len(various),len(various_set))

def get_ncol(data, col):
    result = []
    for i in data:
        result.append(i[col])
    return np.array(result)

def get_ncol_float(data, col):
    result = []
    for i in data:
        result.append(float(i[col]))
    return np.array(result)

min_max_scaler = prepro.MinMaxScaler()
v1 = min_max_scaler.fit_transform(get_ncol_float(cdata,0))
#print v1
#print len(v1)
v2 = category2vec(get_ncol(cdata,1))
v3 = category2vec(get_ncol(cdata,2))
v4 = category2vec(get_ncol(cdata,3))
v5 = category2vec(get_ncol(cdata,4))
v6 = min_max_scaler.fit_transform(get_ncol_float(cdata,5))
v7 = category2vec(get_ncol(cdata,6))
v8 = category2vec(get_ncol(cdata,7))

v9 = category2vec(get_ncol(data,8))
v10 = min_max_scaler.fit_transform(get_ncol_float(data,9))
v11 = category2vec(get_ncol(data,10))
v12 = min_max_scaler.fit_transform(get_ncol_float(data,11))
v13 = min_max_scaler.fit_transform(get_ncol_float(data,12))
v14 = min_max_scaler.fit_transform(get_ncol_float(data,13))
v15 = category2vec(get_ncol(data,14))
y = category2vec(get_ncol(data,15))

trainData = []
for i in range(len(cdata)):
    trainpart=[]
    trainpart.append(v1[i])
    trainpart = trainpart+v2[i].tolist()
    trainpart = trainpart+v3[i].tolist()
    trainpart = trainpart+v4[i].tolist()
    trainpart = trainpart+v5[i].tolist()
    trainpart.append(v6[i])
    trainpart = trainpart+v7[i].tolist()
    trainpart = trainpart+v8[i].tolist()
    trainData.append(trainpart)
    #trainData.append([np.array(v1[i]),v2[i],v3[i],v4[i],v5[i],np.array(v6[i]),v7[i],v8[i]])

trainData = np.array(trainData)
print trainData[0]

#hclust = clust.AgglomerativeClustering(n_clusters=4)
# not enough memory
#clustresult = hclust.fit(trainData)
#kclust = clust.k_means(n_clusters = 4, init='k-means++', precompute_distances= 'auto')
kclust3 = clust.KMeans(n_clusters = 3, random_state=0).fit(trainData)
kclust4 = clust.KMeans(n_clusters = 4, random_state=0).fit(trainData)
kclust5 = clust.KMeans(n_clusters = 5, random_state=0).fit(trainData)
kclust6 = clust.KMeans(n_clusters = 6, random_state=0).fit(trainData)

print kclust4.cluster_centers_

trainData = []
for i in range(len(cdata)):
    trainpart=[]
    trainpart.append(v1[i])
    trainpart = trainpart+v2[i].tolist()
    trainpart = trainpart+v3[i].tolist()
    trainpart = trainpart+v4[i].tolist()
    trainpart = trainpart+v5[i].tolist()
    trainpart.append(v6[i])
    trainpart = trainpart+v7[i].tolist()
    trainpart = trainpart+v8[i].tolist()
    trainpart = trainpart+v9[i].tolist()
    trainpart.append(v10[i])
    trainpart = trainpart+v11[i].tolist()
    trainpart.append(v12[i])
    trainpart.append(v13[i])
    trainpart.append(v14[i])
    trainpart = trainpart+v15[i].tolist()
    trainData.append(trainpart)

trainData = np.array(trainData)
X = trainData
dctree = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, random_state=0)
extree = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(dctree, X, y)
print "decision tree result"
print scores.mean()
#dtree estmater = 10  ---> 0.65
scores = cross_val_score(rf, X, y)
print "random forest tree result"
print scores.mean()
#rf estmater = 10  ---> 0.75
#rf estmater = 50  ---> 0.82
#rf estmater = 100  ---> 0.83
scores = cross_val_score(extree, X, y)
print "extra tree result"
print scores.mean()
#ext estmater = 50  ---> 0.8
#ext estmater = 100  ---> 0.8

#ada = AdaBoostClassifier(n_estimators=100)
#scores = cross_val_score(ada, X, y)
#print "AdaBoost tree result"
#print scores.mean()

