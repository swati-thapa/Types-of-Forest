#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
data=pd.read_csv('iBeacon_RSSI_Labeled.csv')
data.head(2)


# In[114]:


data.drop(['date'],axis=1,inplace=True)


# In[115]:


data.shape


# In[116]:


data.location.unique()


# In[117]:


data.shape


# In[118]:


data1=data.drop_duplicates(subset=data.columns,keep="first")
print(data1)


# In[119]:


data1.b3001.unique()


# In[120]:


data1.b3002.unique()


# In[121]:


data1.b3003.unique()


# In[122]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3001_bin']=pd.cut(data1['b3001'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3001_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)


# In[123]:


data2=pd.crosstab(data1['location'],data1['b3001_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)


# In[124]:


#data2[data2['Strong'] >=10].sort_values(by=['Strong'],ascending=False)


# In[125]:


y=data2.index
x=data2


# In[126]:


import matplotlib.pyplot as plt 
plt.scatter(x,y)
plt.show()


# In[127]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3002_bin']=pd.cut(data1['b3002'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3002_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)


# In[128]:


location_b3002_bin=pd.crosstab(data1['location'],data1['b3002_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)


# In[129]:


x1=location_b3002_bin.index
y1=location_b3002_bin


# In[130]:


plt.scatter(y1,x1)
plt.show()


# In[131]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3003_bin']=pd.cut(data1['b3003'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3003_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3003_bin=pd.crosstab(data1['location'],data1['b3003_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3003_bin
y=location_b3003_bin.index


plt.scatter(x,y)
plt.show()


# In[132]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3004_bin']=pd.cut(data1['b3004'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3004_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3004_bin=pd.crosstab(data1['location'],data1['b3004_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3004_bin
y=location_b3004_bin.index


plt.scatter(x,y)
plt.show()


# In[133]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3004_bin']=pd.cut(data1['b3004'],bins,labels=groups)
#data1['b3004_bin']=pd.cut(data1['b3004'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3004_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3004_bin=pd.crosstab(data1['location'],data1['b3004_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3004_bin
y=location_b3004_bin.index


plt.scatter(x,y)
plt.show()


# In[134]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3005_bin']=pd.cut(data1['b3005'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3005_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3005_bin=pd.crosstab(data1['location'],data1['b3005_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3005_bin
y=location_b3005_bin.index


plt.scatter(x,y)
plt.show()


# In[135]:



bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3006_bin']=pd.cut(data1['b3006'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3006_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3006_bin=pd.crosstab(data1['location'],data1['b3006_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3006_bin
y=location_b3006_bin.index


plt.scatter(x,y)
plt.show()


# In[136]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3007_bin']=pd.cut(data1['b3007'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3007_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3007_bin=pd.crosstab(data1['location'],data1['b3007_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3007_bin
y=location_b3007_bin.index


plt.scatter(x,y)
plt.show()


# In[137]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3008_bin']=pd.cut(data1['b3008'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3008_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3008_bin=pd.crosstab(data1['location'],data1['b3008_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3008_bin
y=location_b3008_bin.index


plt.scatter(x,y)
plt.show()


# In[138]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3009_bin']=pd.cut(data1['b3009'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3009_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3009_bin=pd.crosstab(data1['location'],data1['b3009_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3009_bin
y=location_b3009_bin.index


plt.scatter(x,y)
plt.show()


# In[139]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3010_bin']=pd.cut(data1['b3010'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3010_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3010_bin=pd.crosstab(data1['location'],data1['b3010_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3010_bin
y=location_b3010_bin.index


plt.scatter(x,y)
plt.show()


# In[140]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3011_bin']=pd.cut(data1['b3011'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3011_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3011_bin=pd.crosstab(data1['location'],data1['b3011_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3011_bin
y=location_b3011_bin.index


plt.scatter(x,y)
plt.show()


# In[141]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3012_bin']=pd.cut(data1['b3012'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3012_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3012_bin=pd.crosstab(data1['location'],data1['b3012_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3012_bin
y=location_b3012_bin.index


plt.scatter(x,y)
plt.show()


# In[142]:


bins=[-250,-100,-80,-40,0]
groups=['4','3','2','1']
data1['b3013_bin']=pd.cut(data1['b3013'],bins,labels=groups)
#b3001_bin_location=
pd.crosstab(data1['location'],data1['b3013_bin']).head(10)
#like_o_bin_match.div(like_o_bin_match.sum(1).astype(float),axis=0).plot(figsize=(12,2),kind='line',ax=ax)

location_b3013_bin=pd.crosstab(data1['location'],data1['b3013_bin']).iloc[:,2:].sort_values(by=['2'],ascending=False).head(10)

x=location_b3013_bin
y=location_b3013_bin.index


plt.scatter(x,y)
plt.show()


# In[143]:


data1.iloc[:,14:].astype(int)


# In[144]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data1['location']=label_encoder.fit_transform(data1['location'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)


# In[147]:


data1.iloc[:,14:]=data1.iloc[:,14:].astype(int)


# In[148]:


data1.iloc[:,1:14]


# In[149]:


# Assign varibale and X and y
x = data1.iloc[:,1:14]
y = data1['location']


# In[150]:


# Split
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score,validation_curve,KFold
from sklearn import model_selection
seed = 10
train_x, test_x, train_y, test_y = train_test_split(x,y,random_state=seed, test_size=.5)


# In[151]:


# Import All classifcation Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,Log,Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier,NearestNeighbors, NearestCentroid
from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor


# In[152]:


data1


# In[153]:


#train_x= train_x.values.reshape(1,-1)
#train_y= train_y.values.reshape(1, -1)
#test_x = test_x.values.reshape(1, -1)


# In[154]:


train_x.shape


# In[155]:


train_y.shape


# In[156]:


import sklearn.metrics as metrics


# In[157]:


model = LogisticRegression(C=1, random_state=3)
model.fit(train_x, train_y)
predict_train_lrc = model.predict(train_x)
predict_test_lrc = model.predict(test_x)
print('Training Accuracy:', metrics.accuracy_score(train_y, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(test_y, predict_test_lrc))


# In[158]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# In[159]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 8, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x,y)


# In[160]:


rf_random.best_params_


# In[161]:


rf_random.best_estimator_


# In[162]:


model_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=100, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=800,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
model_RF.fit(train_x, train_y)
predict_train_lrc = model_RF.predict(train_x)
predict_test_lrc = model_RF.predict(test_x)
print('Training Accuracy:', metrics.accuracy_score(train_y, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(test_y, predict_test_lrc))


# In[163]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

accuracy=[]

skf=StratifiedKFold(n_splits=10,random_state=None)
skf.get_n_splits(x,y)
for train_index,test_index in skf.split(x,y):
    print("Train:",train_index,"Validation:",test_index)
    train_x1,test_x1=x.iloc[train_index], x.iloc[test_index]
    train_y1,test_y1=y.iloc[train_index],y.iloc[test_index]
    
    model_RF.fit(train_x1,train_y1)
    prediction=model_RF.predict(test_x1)
    score=accuracy_score(prediction,test_y1)
    accuracy.append(score)
    
print(accuracy)


# In[164]:


import numpy as np
validation=np.array(accuracy).max()
print('Validation Accuracy:',validation)


# In[178]:


result=model_RF.predict([[-21,-200,-90,-87,-77,-90,-100,-34,-45,-55,-80,-90,-88]])


# In[165]:


print(model_RF.predict([[-21,-200,-90,-87,-77,-90,-100,-34,-45,-55,-80,-90,-88]]))


# In[185]:


def get_key(val):
    for key,value in le_name_mapping.items():
        if val==value:
            return key


get_key(result[0])

import numpy as np
validation=np.array(accuracy).max()
print('Validation Accuracy:',validation)

# Saving model to disk
import pickle
pickle.dump(model_RF, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
result=model_RF.predict([[-21,-200,-90,-87,-77,-90,-100,-34,-45,-55,-80,-90,-88]])
print(model_RF.predict([[-21,-200,-90,-87,-77,-90,-100,-34,-45,-55,-80,-90,-88]]))
print(get_key(result[0]))





