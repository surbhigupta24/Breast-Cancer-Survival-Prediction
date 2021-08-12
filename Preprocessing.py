# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:18:12 2020

@author: SURBHI
"""


import pandas as pd
df = pd.DataFrame()
data = pd.read_csv('BREAST.TXT', header = None)

Marital_Status_at_DX = []
for i in data[0]:
    try:
        i = int(i[18])
        Marital_Status_at_DX.append(i)
    except:
        Marital_Status_at_DX.append('')
df['Marital_Status_at_DX'] = Marital_Status_at_DX

Race_Ethnicity = []
for i in data[0]:
    try:
        i = int(i[19:21])
        Race_Ethnicity.append(i)
    except:
        Race_Ethnicity.append('')
df['Race_Ethnicity'] = Race_Ethnicity

Sex = []
for i in data[0]:
    try:
        i = int(i[23])
        Sex.append(i)
    except:
        Sex.append('')
df['Sex'] = Sex

Age_at_diagnosis = []
for i in data[0]:
    try:
        i = int(i[24:27])
        Age_at_diagnosis.append(i)
    except:
        Age_at_diagnosis.append('')
df['Age_at_diagnosis'] = Age_at_diagnosis

Sequence_Number_Central = []
for i in data[0]:
    try:
        i = int(i[34:36])
        Sequence_Number_Central.append(i)
    except:
        Sequence_Number_Central.append('')
df['Sequence_Number_Central'] = Sequence_Number_Central

Primary_Site = []
for i in data[0]:
    try:
        i = str(i[42:46])
        Primary_Site.append(i)
    except:
        Primary_Site.append('')
df['Primary_Site'] = Primary_Site

Histologic_Type_ICD_O_3 = []
for i in data[0]:
    try:
        i = int(i[52:56])
        Histologic_Type_ICD_O_3.append(i)
    except:
        Histologic_Type_ICD_O_3.append('')
df['Histologic_Type_ICD_O_3'] = Histologic_Type_ICD_O_3

Behavior_Code_ICD_O_3 = []
for i in data[0]:
    try:
        i = int(i[56])
        Behavior_Code_ICD_O_3.append(i)
    except:
        Behavior_Code_ICD_O_3.append('')
df['Behavior_Code_ICD_O_3'] = Behavior_Code_ICD_O_3

Grade = []
for i in data[0]:
    try:
        i = int(i[57])
        Grade.append(i)
    except:
        Grade.append('')
df['Grade'] = Grade

Diagnostic_Confirmation = []
for i in data[0]:
    try:
        i = int(i[58])
        Diagnostic_Confirmation.append(i)
    except:
        Diagnostic_Confirmation.append('')
df['Diagnostic_Confirmation'] = Diagnostic_Confirmation

EOD_Tumor_Size = []
for i in data[0]:
    try:
        i = int(i[60:63])
        EOD_Tumor_Size.append(i)
    except:
        EOD_Tumor_Size.append('')
df['EOD_Tumor_Size'] = EOD_Tumor_Size

CS_Tumor_Size = []
for i in data[0]:
    try:
        i = int(i[95:98])
        CS_Tumor_Size.append(i)
    except:
        CS_Tumor_Size.append('')
df['CS_Tumor_Size'] = CS_Tumor_Size

EOD_Extension = []
for i in data[0]:
    try:
        i = int(i[63:65])
        EOD_Extension.append(i)
    except:
        EOD_Extension.append('')
df['EOD_Extension'] = EOD_Extension

CS_Extension = []
for i in data[0]:
    try:
        i = int(i[98:101])
        CS_Extension.append(i)
    except:
        CS_Extension.append('')
df['CS_Extension'] = CS_Extension

EOD_Lymph_Node_Involv = []
for i in data[0]:
    try:
        i = int(i[67])
        EOD_Lymph_Node_Involv.append(i)
    except:
        EOD_Lymph_Node_Involv.append('')
df['EOD_Lymph_Node_Involv'] = EOD_Lymph_Node_Involv

CS_Lymph_Nodes = []
for i in data[0]:
    try:
        i = int(i[101:104])
        CS_Lymph_Nodes.append(i)
    except:
        CS_Lymph_Nodes.append('')
df['CS_Lymph_Nodes'] = CS_Lymph_Nodes

Regional_Nodes_Positive = []
for i in data[0]:
    try:
        i = int(i[68:70])
        Regional_Nodes_Positive.append(i)
    except:
        Regional_Nodes_Positive.append('')
df['Regional_Nodes_Positive'] = Regional_Nodes_Positive

Regional_Nodes_Examined = []
for i in data[0]:
    try:
        i = int(i[70:72])
        Regional_Nodes_Examined.append(i)
    except:
        Regional_Nodes_Examined.append('')
df['Regional_Nodes_Examined'] = Regional_Nodes_Examined

Reason_for_no_surgery = []
for i in data[0]:
    try:
        i = int(i[165])
        Reason_for_no_surgery.append(i)
    except:
        Reason_for_no_surgery.append('')
df['Reason_for_no_surgery'] = Reason_for_no_surgery

SEER_historic_stage_A = []
for i in data[0]:
    try:
        i = int(i[235])
        SEER_historic_stage_A.append(i)
    except:
        SEER_historic_stage_A.append('')
df['SEER_historic_stage_A'] = SEER_historic_stage_A

First_malignant_primary_indicator = []
for i in data[0]:
    try:
        i = int(i[244])
        First_malignant_primary_indicator.append(i)
    except:
        First_malignant_primary_indicator.append('')
df['First_malignant_primary_indicator'] = First_malignant_primary_indicator

Cause_of_Death_to_SEER_site_recode = []
for i in data[0]:
    try:
        i = int(i[254:259])
        Cause_of_Death_to_SEER_site_recode.append(i)
    except:
        Cause_of_Death_to_SEER_site_recode.append('')
df['Cause_of_Death_to_SEER_site_recode'] = Cause_of_Death_to_SEER_site_recode

Vital_Status_recode = []
for i in data[0]:
    try:
        i = int(i[264])
        Vital_Status_recode.append(i)
    except:
        Vital_Status_recode.append('')
df['Vital_Status_recode'] = Vital_Status_recode

Survival_months = []
for i in data[0]:
    try:
        i = int(i[300:304])
        Survival_months.append(i)
    except:
        Survival_months.append('')
df['Survival_months'] = Survival_months

df.to_csv('bb.csv')

import pandas as pd
data  = pd.read_csv('bb.csv')
data = data.iloc[:, 1:]

vals = []
for i in data.columns:
    a = data[i].value_counts()
    vals.append(a)
cols = list(data.columns)

percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})

data = data.drop(columns = ['EOD_Tumor_Size', 'EOD_Extension', 'EOD_Lymph_Node_Involv'])
data.to_csv('bb1.csv')

################ preprocessing few columns

import pandas as pd
import numpy as np
df = pd.read_csv('bb1.csv')
df1 = df.iloc[:, 1:]
cols1 = list(df1.columns)

vals = []
for i in df1.columns:
    a = df1[i].value_counts()
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

df3.to_csv('bb_1.csv')


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from numpy import unique
from imblearn.combine import SMOTEENN

df = pd.read_csv('bb_1.csv')
df = df.iloc[:, 1:]

df['SurvivalMonths_VitalStatus'].value_counts()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

SENN = SMOTEENN()
X, y = SENN.fit_sample(X, y)

X = X.reshape(X.shape[0], X.shape[1], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(19,1)))
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

########################### BernoulliRBM

import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN

df = pd.read_csv('bb_1.csv')
df = df.iloc[:, 1:]

df['SurvivalMonths_VitalStatus'].value_counts()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

SENN = SMOTEENN()
X, y = SENN.fit_sample(X, y)

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