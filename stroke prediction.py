#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 10)


# In[11]:


data=pd.read_csv(r'C:\Users\hp\Downloads\Stroke-Risk-Prediction-using-Machine-Learning-master\Stroke-Risk-Prediction-using-Machine-Learning-master\dataset\healthcare-dataset-stroke-data.csv')


# In[12]:


data


# exploring data

# In[13]:


data.info()


# In[14]:


data.isnull().sum()


# lets fill null values

# In[15]:


data['bmi'].value_counts()


# In[16]:


data['bmi'].describe()


# In[22]:


data['bmi'].fillna(data['bmi'].mean(),inplace=True)


# In[23]:


data.isnull().sum()


# In[25]:


data.drop('id',axis=1,inplace=True)


# In[26]:


data


# #outlier Removation

# In[29]:


data.plot(kind='box')
plt.show()


# In[30]:


data['avg_glucose_level'].describe()


# In[33]:


data[data['avg_glucose_level']>114.090000]


# #Label Encoding
# 

# In[41]:


data.head()
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[42]:


gender=enc.fit_transform(data['gender'])


# In[43]:


smoking_status=enc.fit_transform(data['smoking_status'])


# In[44]:


work_type=enc.fit_transform(data['work_type'])
Residence_type=enc.fit_transform(data['Residence_type'])
ever_married=enc.fit_transform(data['ever_married'])


# In[45]:


data['ever_married']=ever_married
data['Residence_type']=Residence_type
data['work_type']=work_type
data['smoking_status']=smoking_status
data['gender']=gender


# In[46]:


data


# In[47]:


data.info()


# #splitting the data for train and test (partioning)

# X----train_X,test_X 80/20
# Y----train_Y,test_Y

# In[52]:


X=data.drop('stroke',axis=1)
X.head()
Y=data['stroke']
Y


# In[53]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


# In[59]:


X_train


# In[60]:


X_test


# In[61]:


Y_test


# #Normalize

# In[63]:


data.describe()


# In[80]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[82]:


X_train_std=std.fit_transform(X_train)


# In[83]:


X_test_std=std.transform(X_test)


# In[84]:


X_train_std


# In[85]:


X_test_std


# #Training

# #applying algorithms now

# # DECISION TREE

# In[88]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[90]:


dt.fit(X_train_std,Y_train)


# In[91]:


dt.feature_importances_


# In[93]:


X_train.columns


# In[102]:


Y_pred=dt.predict(X_test_std)
Y_pred


# In[95]:


Y_test


# In[96]:


from sklearn.metrics import accuracy_score


# In[103]:


ac_dt=accuracy_score(Y_test,Y_pred)


# In[104]:


ac_dt


# # LOgistic Regression

# In[105]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[106]:


LogisticRegression()


# In[107]:


lr.fit(X_train_std,Y_train)


# In[108]:


Y_pred_lr=lr.predict(X_test_std)


# In[109]:


Y_pred_lr


# In[110]:


ac_lr=accuracy_score(Y_test,Y_pred_lr)


# In[111]:


ac_lr


# # KNN algorithm

# In[113]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[114]:


KNeighborsClassifier()


# In[115]:


knn.fit(X_train_std,Y_train)


# In[116]:


Y_pred=knn.predict(X_test_std)


# In[117]:


ac_knn=accuracy_score(Y_test,Y_pred)


# In[118]:


ac_knn


# # Random Forest

# In[119]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[120]:


RandomForestClassifier()


# In[121]:


rf.fit(X_train_std,Y_train)


# In[122]:


Y_pred=rf.predict(X_test_std)


# In[123]:


ac_rf=accuracy_score(Y_test,Y_pred)


# In[124]:


ac_rf


# In[125]:


ac_knn


# # SVM

# In[127]:


from sklearn.svm import SVC
sv=SVC()


# In[128]:


SVC()


# In[129]:


sv.fit(X_train_std,Y_train)


# In[130]:


Y_pred=sv.predict(X_test_std)


# In[131]:


ac_sv=accuracy_score(Y_test,Y_pred)


# In[132]:


ac_sv


# In[133]:


ac_lr


# In[134]:


data


# In[139]:


plt.bar(['Decision Tree','Logistic','KNN','Random Forest','SVM'],[ac_dt,ac_lr,ac_knn,ac_rf,ac_sv])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()


# In[141]:


import pickle
filename = r'C:\Users\hp\Downloads\Stroke-Risk-Prediction-using-Machine-Learning-master\Stroke-Risk-Prediction-using-Machine-Learning-master\finalized_model_lr.sav'
pickle.dump(lr,open(filename, 'wb'))


# In[ ]:




