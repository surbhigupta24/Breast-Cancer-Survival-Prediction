# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:53:02 2020

@author: Rishav
"""

import pandas as pd

df = pd.DataFrame()
data = pd.read_csv('BREAST.TXT', header = None)

Primary_Site = []
for i in data[0]:
    try:
        i = int(i[42:46])
        Primary_Site.append(i)
    except:
        Primary_Site.append('')
df['Primary_Site'] = Primary_Site

df.to_csv('Breast.csv')

#CS Mets at Dx-Bone
#CS Mets at Dx-Brain
#CS Mets at Dx-Liver
#CS Mets at Dx-Lung
#EOD—Old 13 Digit
#EOD—Old 2 Digit
#EOD—Old 4 Digit
#Coding System for EOD
#Tumor Marker 1
#Tumor Marker 2
#Tumor Marker 3
#RX Summ—Reg LN Examined
#RX Summ—Surgery Type
#RX Summ—Scope Reg 98-02
#RX Summ—Surg Oth 98-02
#SEER Summary Stage 1977 (1995-2000)
#SEER Summary Stage 2000 (2001-2003)
#Insurance recode (2007+)
#Derived AJCC-7 T
#Derived AJCC-7 N
#Derived AJCC-7 M
#Derived AJCC-7 Stage Grp
#Derived HER2 Recode (2010+)
#Breast Subtype (2010+)

################ for adding primary site column

import pandas as pd
data = pd.read_csv('Breast.csv')
data1 = pd.read_csv('BREAST.TXT', header = None)

Primary_Site = []
for i in data1[0]:
    try:
        i = str(i[42:46])
        Primary_Site.append(i)
    except:
        Primary_Site.append('')
data['Primary_Site'] = Primary_Site
data.to_csv('Breast.csv')


################################################

import pandas as pd
data  = pd.read_csv('Breast.csv')
data = data.iloc[:, 2:]

vals = []
for i in data.columns:
    a = data[i].value_counts()
    vals.append(a)
    
cols = list(data.columns)
# removing data columns according to sames value, nan value, and all different values
data1 = data.drop(columns = ['Patient_ID_number', 'EOD_Extension_Prost_Path',
                             'CS_Site_Specific_Factor25', 'Derived_AJCC_Flag',
                             'Derived_AJCC_Flag', 'RX_Summ_Scope_Reg_LN_Sur', 'Site_Recode_ICD_O_3_WHO_2008',
                             'Recode_ICD_O_2_to_10', 'Histology_Recode_Brain_Groupings',
                             'CS_Schema_v0204_plus', 'Lymphoma_subtype_recode_WHO_2008', 
                             'CS_Schema_AJCC_6thed_previously_called_v1', 'CS_Site_Specific_Factor8',
                             'CS_Site_Specific_Factor10', 'CS_Site_Specific_Factor11', 'CS_Site_Specific_Factor13',
                             'CS_Site_Specific_Factor16', 'Lymph_vascular_invasion', 'CS_Site_Specific_Factor_9',
                             'CS_Site_Specific_Factor_12', 'Lymphomas_Ann_Arbor_Staging_1983_plus'])


percent_missing = data1.isnull().sum() * 100 / len(data1)
missing_value_df = pd.DataFrame({'column_name': data1.columns,
                                 'percent_missing': percent_missing})
    
cols1 = list(data1.columns)
#removing columns that have more than 50% missing values
data2 = data1.drop(columns = ['EOD_Tumor_Size', 'EOD_Extension', 'EOD_Lymph_Node_Involv', 
                              'AJCC_stage_3rd_edition_1988_2003', 'SEER_modified_AJCC_stage_3rd_edition_1988_2003',
                              'CS_Site_Specific_Factor15', 'CS_Site_Specific_Factor_7', 
                              'T_value_based_on_AJCC_3rd', 'N_value_based_on_AJCC_3rd',
                              'M_value_based_on_AJCC_3rd'])

    
#removing rows that have missing values
data3 = data2.dropna()
percent_missing = data3.isnull().sum() * 100 / len(data3)
missing_value_df = pd.DataFrame({'column_name': data3.columns,
                                 'percent_missing': percent_missing})
data3.to_csv('Breast_cleaned.csv')

################ preprocessing few columns

import pandas as pd
import numpy as np
df = pd.read_csv('Breast_cleaned.csv')
df1 = df.iloc[:, 1:]
cols1 = list(df.columns)

vals = []
for i in df.columns:
    a = df[i].value_counts()
    vals.append(a)

df1['Survival_months'] = df1['Survival_months'].apply(lambda x: 1 if x >= 60 else 0)
df1['Vital_Status_recode'] = df1['Vital_Status_recode'].apply(lambda x: 0 if x != 1 else 1)
df1['Cause_of_Death_to_SEER_site_recode'] = df1['Cause_of_Death_to_SEER_site_recode'].replace(0, 1)
df1['Cause_of_Death_to_SEER_site_recode'] = df1['Cause_of_Death_to_SEER_site_recode'].replace(26000 , 0)
df1['Cause_of_Death_to_SEER_site_recode'] = df1['Cause_of_Death_to_SEER_site_recode'].apply(lambda x: np.nan if (x != 1 and x != 0) else x)

df2 = df1.dropna()
    
cols1 = list(df2.columns)
vals = []
for i in df2.columns:
    a = df2[i].value_counts()
    vals.append(a)
    
percent_missing = df2.isnull().sum() * 100 / len(df2)

df2.loc[(df2['Survival_months'] == 1) & (df2['Vital_Status_recode'] == 1), 'SurvivalMonths_VitalStatus'] = 1
df2.loc[(df2['Survival_months'] != 1) | (df2['Vital_Status_recode'] != 1), 'SurvivalMonths_VitalStatus'] = 0

df3 = df2.drop(columns = ['Survival_months', 'Vital_Status_recode'])

df3['Primary_Site'] = df3['Primary_Site'].map({'C504':4, 'C508':8, 'C509':9, 'C502':4, 'C505':5,
                                  'C501':1, 'C503':3, 'C500':0, 'C506':6})

df3.to_csv('Breast_cleaned_1.csv')

########################### BernoulliRBM

import pandas as pd
import numpy as np

df = pd.read_csv('Breast_cleaned_1.csv')
df = df.iloc[:, 1:]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn import linear_model
from sklearn import metrics

logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=21, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])

rbm.learning_rate = 0.06
rbm.n_iter = 10

rbm.n_components = 100
logistic.C = 6000

rbm_features_classifier.fit(X_train, y_train)

raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.
raw_pixel_classifier.fit(X_train, y_train)

Y_pred = rbm_features_classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, Y_pred)))

metrics.confusion_matrix(y_test,Y_pred)

Y_pred = raw_pixel_classifier.predict(X_test)
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(y_test, Y_pred)))
metrics.accuracy_score(Y_pred, y_test)
## accuracy = 0.9738743369182579


########################### CONV1D

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import unique

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

X = X.reshape(X.shape[0], X.shape[1], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(77,1)))
model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', 
     optimizer = "rmsprop",               
              metrics = ['accuracy'])
model.summary()
    
model.fit(X_train, y_train, batch_size=100,epochs=20, verbose=0)
acc = model.evaluate(X_train, y_train)

pred = np.round(model.predict(X_test))
pred1 = pd.Series(pred[:, 0])

cm = confusion_matrix(y_test, pred1)
cr = metrics.classification_report(y_test, pred1)
## accuracy 92

################### Deep AutoEncoders

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn import metrics

df = pd.read_csv('Breast_cleaned_1.csv')
df = df.iloc[:, 1:]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

input_data= Input(shape=(77,))
encoded = Dense(units=64, activation='relu')(input_data)
encoded = Dense(units=32, activation='relu')(encoded)
encoded = Dense(units=16, activation='relu')(encoded)
decoded = Dense(units=32, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(decoded)
decoded = Dense(units=77, activation='relu')(decoded)
decoded = Dense(units=1, activation='sigmoid')(decoded)
autoencoder=Model(input_data, decoded)
encoder = Model(input_data, encoded)
autoencoder.summary()
encoder.summary()

autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X_train, y_train,
                epochs=5,
                batch_size=200,
                shuffle=True,
                validation_data=(X_test, y_test))

pred = np.round(autoencoder.predict(X_test))
pred1 = pd.Series(pred[:, 0])

cm = metrics.confusion_matrix(y_test, pred1)
cr = metrics.classification_report(y_test, pred1)
## accuracy 92.31






