# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:28:16 2021

@author: Aman Gupta
"""

import pandas as pd
import xgboost
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('train_Df64byy.csv')
X = df.iloc[:,1:13]
Y = df.iloc[:,-1]
a0 = 0
a1 = 0
for i in Y:
    if i ==0:
        a0 +=1
    else:
        a1 += 1

df_test = pd.read_csv('test_YCcRUnU.csv')
x = df_test.iloc[:,1:]
test_member_id = df_test.iloc[:,:1]


##############  Data Transformation#########
labelEncoder= LabelEncoder()
X['City_Code'] = labelEncoder.fit_transform(X['City_Code'])
x['City_Code'] = labelEncoder.fit_transform(x['City_Code'])


print('Holding_Policy_Duration...')
X['Holding_Policy_Duration'].replace(to_replace='\+', value='', regex=True, inplace=True)
X['Holding_Policy_Duration'] = pd.to_numeric(X['Holding_Policy_Duration'], errors='coerce')
x['Holding_Policy_Duration'].replace(to_replace='\+', value='', regex=True, inplace=True)
x['Holding_Policy_Duration'] = pd.to_numeric(x['Holding_Policy_Duration'], errors='coerce')

print('Accomodation_Type...')
X['Accomodation_Type'].replace(to_replace='Rented', value='0', regex=True, inplace=True)
X['Accomodation_Type'].replace(to_replace='Owned', value='1', regex=True, inplace=True)
x['Accomodation_Type'].replace(to_replace='Rented', value='0', regex=True, inplace=True)
x['Accomodation_Type'].replace(to_replace='Owned', value='1', regex=True, inplace=True)

print('Reco_Insurance_Type...')
X['Reco_Insurance_Type'].replace(to_replace='Individual', value='0', regex=True, inplace=True)
X['Reco_Insurance_Type'].replace(to_replace='Joint', value='1', regex=True, inplace=True)
x['Reco_Insurance_Type'].replace(to_replace='Individual', value='0', regex=True, inplace=True)
x['Reco_Insurance_Type'].replace(to_replace='Joint', value='1', regex=True, inplace=True)

print('Is_Spouse...')
X['Is_Spouse'].replace(to_replace='Yes', value='0', regex=True, inplace=True)
X['Is_Spouse'].replace(to_replace='No', value='1', regex=True, inplace=True)
x['Is_Spouse'].replace(to_replace='Yes', value='0', regex=True, inplace=True)
x['Is_Spouse'].replace(to_replace='No', value='1', regex=True, inplace=True)

hi = X['Health Indicator'].unique()
print('Health Indicator...')
X['Health Indicator'].replace(to_replace='X1', value='0', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X2', value='1', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X3', value='2', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X4', value='3', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X5', value='4', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X6', value='5', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X7', value='6', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X8', value='7', regex=True, inplace=True)
X['Health Indicator'].replace(to_replace='X9', value='8', regex=True, inplace=True)

x['Health Indicator'].replace(to_replace='X1', value='0', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X2', value='1', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X3', value='2', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X4', value='3', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X5', value='4', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X6', value='5', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X7', value='6', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X8', value='7', regex=True, inplace=True)
x['Health Indicator'].replace(to_replace='X9', value='8', regex=True, inplace=True)

########### Missing values  ###########
cols = ['Holding_Policy_Duration','Holding_Policy_Type','Health Indicator']
for col in cols:
    print('Imputation with Median: %s' % (col))
    X[col].fillna(X[col].median(), inplace=True)
    x[col].fillna(X[col].median(), inplace=True)
    #X[col].fillna(0, inplace=True)
    #x[col].fillna(0, inplace=True)
	
from imblearn.over_sampling import RandomOverSampler
randomsample=  RandomOverSampler()
x_new,y_new=randomsample.fit_resample(X,Y)

scaled_features = StandardScaler().fit_transform(x_new.values)
#Split train and cross validation sets
X_train, X_test, y_train, y_test = train_test_split(np.array(scaled_features), np.array(y_new), test_size=0.30)
eval_set=[(X_test, y_test)]



clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=eval_set, verbose=True)


submission_file_name = 'xgb_sol_rss_scaled'
y_pred = clf.predict(np.array(X_test))
accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
submission_file_name = submission_file_name + ("_Accuracy_%.6f" % (accuracy * 100)) + '_'

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
submission_file_name = submission_file_name + ("_ROC-AUC_%.6f" % (accuracy_per_roc_auc * 100))


xscaled_features = StandardScaler().fit_transform(x.values)
final_pred = pd.DataFrame(clf.predict_proba(np.array(x)))
dfSub = pd.concat([test_member_id, final_pred.loc[:, 1:2]], axis=1)
dfSub.rename(columns={1:'Response'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)









