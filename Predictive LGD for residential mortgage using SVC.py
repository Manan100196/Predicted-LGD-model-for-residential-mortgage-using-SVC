#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data = pd.read_csv(r'E:\Credit Risk Management\Project\ArizonaCalifornia_Final.csv')


# In[2]:


data.head()


# In[3]:


# Checked the unique values in the state column
data.STATE.unique()


# In[4]:


# Filtered California data
data = data[data['STATE'] == 'CA']


# In[5]:


# Drop columns which were not required
data.drop(['VAR1','LOAN_ID','ORIG_RATE','CURR_RATE','ORIG_DATE','OCLTV','OCC_STAT','STATE','MSA','ZIP','MI_PCT','Zero_Bal_Code.1','LAST_UPB','LAST_ACTIVITY_DATE','Unemp_Orig','Unemp_Monthly_Orig','Unemp_Yearly_Orig','Unemp_Monthly_Last','Unemp_Yearly_Last','HPI','HPI_Quarterly','HPI_Yearly','HPI_Quarterly_Last','HPI_Yearly_Last','Date_Origination','Date_Last_Activity'], inplace = True, axis = 1)


# In[6]:


# Checked the info of the dataframe
data.info()


# In[7]:


data


# In[8]:


#1046 rows were dropped which had na values
data = data.dropna()
data


# In[9]:


#Loss severity values less that 0 is updated as 0
#Loss severity values greater than 1 is updated as 1
for i in range(data.shape[0]):
    if data.iloc[i,17] < 0:
        data.iloc[i,17] = 0
    elif data.iloc[i,17] > 1:
        data.iloc[i,17] = 1


# The dataset needs to be classified into three parts
# 1 - Loans with no loss
# 2 - Loans with partial loss
# 3 - Loans wih complete loss
# 
# Since, SVC is binary classification, we will classify the data into 
# 1 - Loans with no loss
# 2 - Loans with loss
# 
# The loans which had losses, that is the second one is further classified into 
# 1 - Loans with partial loss
# 2 - Loans with complete loss

# In[10]:


# Our aim is to classify data set into two parts
# A new column name Upd_Loss_Sev (updated Loss Severity) is created and updated based on value in Loss_Sev field
# 1. default with no loss (Upd_Loss_Sev = 0)
# 2. default with loss (Upd_Loss_Sev = 1)
data['Upd_Loss_Sev'] = 0
for i in range(data.shape[0]):
    if data.iloc[i,17] != 0:
        data.iloc[i,20] = 1
    else:
        data.iloc[i,20] = 0


# In[11]:


# count of data with 0 and 1 values in the field Upd_Loss_Sev
data['Upd_Loss_Sev'].value_counts()


# In[12]:


# Creating input dataset
X = data.drop(['Upd_Loss_Sev','Loss_Sev'],axis = 1)


# In[13]:


# Creating output dataset
y = data['Upd_Loss_Sev']


# In[14]:


# splitting the entire dataset into train and test in the ratio 0.8:0.2. Later, the input dataset X is scaled

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)


# In[15]:


#Performing support vector classification on training dataset. Later, calculating accuracy, precision and recall for
# entire dataset

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, plot_confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred = svc.predict(X_Scaled)

print(f'Accuracy = {accuracy_score(y, pred):.2f}\nRecall = {recall_score(y, pred):.2f}\nPrecision = {precision_score(y, pred):.2f}')
cm = confusion_matrix(y, pred)
plt.figure()
plt.title('Confusion Matrix on entire dataset')
sns.heatmap(cm, annot = True)


# In[16]:


#Performing support vector classification on train dataset. Later, calculating accuracy, precision and recall on training set

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, plot_confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_train = svc.predict(X_train)

print(f'Accuracy_train = {accuracy_score(y_train, pred_train):.2f}\nRecall_train = {recall_score(y_train, pred_train):.2f}\nPrecision_train = {precision_score(y_train, pred_train):.2f}')
cm = confusion_matrix(y_train, pred_train)
plt.figure()
plt.title('Confusion Matrix on training dataet')
sns.heatmap(cm, annot = True)


# In[17]:


# Performing support vector classification on train dataset. Later, calculating accuracy, precision and recall on training set
# comparing result with training set done above to check for variance

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, plot_confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_test = svc.predict(X_test)

print(f'Accuracy_test = {accuracy_score(y_test, pred_test):.2f}\nRecall_test = {recall_score(y_test, pred_test):.2f}\nPrecision_test = {precision_score(y_test, pred_test):.2f}')
cm = confusion_matrix(y_test, pred_test)
plt.figure()
plt.title('Confusion Matrix on testing dataset')
sns.heatmap(cm, annot = True)


# In[18]:


# Updating the dataset with new column "Pred[No Loss and partial loss]" based on SVC model created above

data['Pred[No Loss and partial loss]'] = pred


# In[19]:


data


# In[20]:


data['Pred[No Loss and partial loss]'].value_counts()


# From here, we will start second classification. That is, classifying loans with losses into 1 - Loans with partial loss and 2 - loans with complete loss

# In[21]:


# For second classification, filtering only those loans which were predicted to have loss

data = data[data['Pred[No Loss and partial loss]'] == 1]


# In[22]:


data


# In[23]:


# Updated the filtered data with 0 and 1 based on loans with partial loss and loans with complete loss
# 0 - loans with partial loss
# 1 - loans with complete loss

for i in range(data.shape[0]):
    if data.iloc[i,17] < 1:
        #print(data.iloc[i,17])
        data.iloc[i,20] = 0
    #elif data.iloc[i,17] = 1:
    else:
        data.iloc[i,20] = 1


# In[24]:


#Count of loans with partial loss (0) and Full Loss(1)

data['Upd_Loss_Sev'].value_counts()


# In[25]:


# We run the SVC again and figure it out that recall and precision score are 0. Hence, we use smote analysis in the next step.

X = data.drop(['Loss_Sev', 'Upd_Loss_Sev', 'Pred[No Loss and partial loss]'], axis = 1)
y = data['Upd_Loss_Sev']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred = svc.predict(X_test_Scaled)

print(f'Accuracy = {accuracy_score(y_test, pred):.2f} \nRecall = {recall_score(y_test, pred):.2f} \nPrecision = {precision_score(y_test, pred):.2f} \n')
cm = confusion_matrix(y_test, pred)
plt.figure()
plt.title('Confusion Metrics')
sns.heatmap(cm, annot = True)


# In[26]:


# SMOTE ANALYSIS

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)

X_sm, y_sm = sm.fit_resample(X, y)

print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
print(y_sm.value_counts(normalize=True) * 100)


# In[27]:


# SMOTE analysis with SVC. Prediction done on the entire dataset which has randomly generated data of SMOTE as well

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X_sm)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred = svc.predict(X_Scaled)

print(f'Accuracy = {accuracy_score(y_sm, pred):.2f} \nRecall = {recall_score(y_sm, pred):.2f} \nPrecision = {precision_score(y_sm, pred):.2f} \n')
cm = confusion_matrix(y_sm, pred)
plt.figure()
plt.title('Confusion Metrics')
sns.heatmap(cm, annot = True)


# In[28]:


# SMOTE analysis with SVC. Prediction done on the training dataset

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_train = svc.predict(X_train_Scaled)

print(f'Accuracy = {accuracy_score(y_train, pred_train):.2f} \nRecall = {recall_score(y_train, pred_train):.2f} \nPrecision = {precision_score(y_train, pred_train):.2f} \n')
cm_2 = confusion_matrix(y_train, pred_train)
plt.figure()
plt.title('Confusion Metrics')
sns.heatmap(cm_2, annot = True)


# In[29]:


# SMOTE analysis with SVC. Prediction done on the testing dataset
# This is done to check for variance

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_test = svc.predict(X_test_Scaled)

print(f'Accuracy = {accuracy_score(y_test, pred_test):.2f} \nRecall = {recall_score(y_test, pred_test):.2f} \nPrecision = {precision_score(y_test, pred_test):.2f} \n')
cm_1 = confusion_matrix(y_test, pred_test)
plt.figure()
plt.title('Confusion Metrics')
sns.heatmap(cm_1, annot = True)


# In[30]:


# SMOTE result on original dataset

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_orig = svc.predict(X_Scaled)

print(f'Accuracy = {accuracy_score(y, pred_orig):.2f} \nRecall = {recall_score(y, pred_orig):.2f} \nPrecision = {precision_score(y, pred_orig):.2f} \n')
cm_1 = confusion_matrix(y, pred_orig)
plt.figure()
plt.title('SMOTE Confusion Metrics on original dataset')
sns.heatmap(cm_1, annot = True)


# The results with SMOTE were better on SMOTE data but not on original data. Hence, we use down-sample data.  

# In[31]:


#Count of loans with partial loss (0) and Full Loss(1)

data['Upd_Loss_Sev'].value_counts()


# In[32]:


from sklearn.utils import resample

data_downsample_0 = resample(data[data['Upd_Loss_Sev'] == 0], replace = False, n_samples = 154, random_state = 42)
data_downsample_1 = resample(data[data['Upd_Loss_Sev'] == 1], replace = False, n_samples = 154, random_state = 42)


# In[33]:


data_downsample = pd.concat([data_downsample_1, data_downsample_0])
data_downsample.shape


# In[34]:


#Entire dataset result

X = data_downsample.drop(['Upd_Loss_Sev','Loss_Sev'],axis = 1)
y = data_downsample['Upd_Loss_Sev']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred = svc.predict(X_Scaled)

print(f'Accuracy = {accuracy_score(y, pred):.2f}\nRecall = {recall_score(y, pred):.2f}\nPrecision = {precision_score(y, pred):.2f}')
cm = confusion_matrix(y, pred)
plt.figure()
plt.title('Confusion Matrix on entire dataset')
sns.heatmap(cm, annot = True)


# In[35]:


#Training dataset result

X = data_downsample.drop(['Upd_Loss_Sev','Loss_Sev'],axis = 1)
y = data_downsample['Upd_Loss_Sev']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_train = svc.predict(X_train_Scaled)

print(f'Accuracy = {accuracy_score(y_train, pred_train):.2f}\nRecall = {recall_score(y_train, pred_train):.2f}\nPrecision = {precision_score(y_train, pred_train):.2f}')
cm = confusion_matrix(y_train, pred_train)
plt.figure()
plt.title('Confusion Matrix on training dataset')
sns.heatmap(cm, annot = True)


# In[36]:


#Testing dataset result
#Comparing training with testing, we figure it out that there is no variance

#Entire dataset result

X = data_downsample.drop(['Upd_Loss_Sev','Loss_Sev'],axis = 1)
y = data_downsample['Upd_Loss_Sev']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_Scaled = scale(X_train)
X_test_Scaled = scale(X_test)
X_Scaled = scale(X)

svc = SVC(random_state = 42)
svc.fit(X_train_Scaled, y_train)
pred_test = svc.predict(X_test_Scaled)

print(f'Accuracy = {accuracy_score(y_test, pred_test):.2f}\nRecall = {recall_score(y_test, pred_test):.2f}\nPrecision = {precision_score(y_test, pred_test):.2f}')
cm = confusion_matrix(y_test, pred_test)
plt.figure()
plt.title('Confusion Matrix on test dataset')
sns.heatmap(cm, annot = True)

Down-sample analysis result had better accuracy, recall and precision value on original data as well as the test data. Hence, we use the data from down-sample analysis to predict partial losses of the loans using beta regression.  