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

df.to_csv('Breastt.csv')

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
data = pd.read_csv('Breastt.csv')
data1 = pd.read_csv('BREAST.TXT', header = None)

Primary_Site = []
for i in data1[0]:
    try:
        i = str(i[42:46])
        Primary_Site.append(i)
    except:
        Primary_Site.append('')
data['Primary_Site'] = Primary_Site
data.to_csv('Breastt.csv')


################################################

import pandas as pd
data  = pd.read_csv('Breastt.csv')
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

########################### Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Breast_cleaned_1.csv')
df1 = df.iloc[:, 1:]
cols1 = list(df.columns)

vals = []
for i in df1.columns:
    a = df1[i].value_counts()
    vals.append(a)
    
df1[['Vital_Status_recode', 'Survival_months']].groupby('Vital_Status_recode').count()
df1[['Vital_Status_recode', 'Survival_months']].groupby('Vital_Status_recode').sum()
df1[['Vital_Status_recode', 'Survival_months']].groupby('Vital_Status_recode').mean()

plt1 = df1[['Survival_months', 'Vital_Status_recode']].groupby('Vital_Status_recode').mean().Survival_months.plot(kind='bar')
plt1.set_xlabel('Survival_months')
plt1.set_ylabel('Vital_Status_recode')

plt1 = df1[['Marital_Status_at_DX', 'Vital_Status_recode']].groupby('Marital_Status_at_DX').mean().Vital_Status_recode.plot('bar')
plt1.set_xlabel('Marital_Status_at_DX')
plt1.set_ylabel('Vital_Status_recode')

plt1 = df1[['Vital_Status_recode', 'Race_Ethnicity']].groupby('Race_Ethnicity').sum().Vital_Status_recode.plot('bar')
plt1.set_xlabel('Race_Ethnicity')
plt1.set_ylabel('Vital_Status_recode')


#primary state not involved it removed in preprocessing stage
#df1 = df[['Marital_Status_at_DX', 'Race_Ethnicity', 'Age_at_diagnosis', 'SEER_historic_stage_A',
#          'Sequence_Number_Central', 'Histologic_Type_ICD_O_3', 'Behavior_Code_ICD_O_3','Primary_Site',
#          'Grade', 'Diagnostic_Confirmation', 'CS_Tumor_Size', 'CS_Extension', 'CS_Lymph_Nodes',
#          'Regional_Nodes_Positive', 'Regional_Nodes_Examined', 'Reason_for_no_surgery',
#          'First_malignant_primary_indicator', 'Vital_Status_recode', 'Survival_months',
#          'Cause_of_Death_to_SEER_site_recode']]


















