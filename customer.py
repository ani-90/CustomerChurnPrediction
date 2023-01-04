#!/usr/bin/env python
# coding: utf-8

# In[265]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('D:/data science/kaggle_datasets/customerchurn/data.csv.txt')
df.head()


# In[266]:


# pip install statsmodels


# In[ ]:





# In[ ]:





# In[267]:


'''Description of the data:'''
def description(df):
    print("Description of data( an overview)")
    print("No of data points available",df.shape[0])
    print("No of features in the dataset",df.shape[1])
    print()
    print("Other noteworthy points: ")
    print("Missing/null values",df.isnull().sum().sum())
    print("Features of the dataset: ",list(df.columns))
    print("Target variable:",'Churn..(yes or no)\n')
    print('Unique values across each feature(count)\n',df.nunique())


# In[268]:


description(df)


# ### there are 17 categorical and 3 numerical features

# In[ ]:





# # Exploratory Data Analysis

# In[269]:


'''to check what percentage of the customers have actually churned or not'''

churnData=df['Churn'].value_counts().to_frame()
churnData=churnData.reset_index()
churnData=churnData.rename(columns={'index':'Category'})
plot=px.pie(churnData,values='Churn',names='Category',color_discrete_sequence=['green','red'],title='Distribution of Churn')
plot.show()


# In[270]:


'''exploratory data analysis using the categorical variables'''


# In[271]:


def eda_cat(feature,df):
    currentfeature=df.groupby([feature,'Churn']).size().reset_index()
    currentfeature=currentfeature.rename(columns={0:'Count'})
    
    
    distribution=df[feature].value_counts().to_frame().reset_index()
    
    categories=[c[1][0] for c in distribution.iterrows()]
    
    figure=px.bar(currentfeature,x=feature,y='Count',color='Churn',barmode="group", color_discrete_sequence=["green", "red"],title=f'Churn rate by  {feature}')
    figure.show()
    


# In[ ]:





# In[272]:


df.loc[df.SeniorCitizen==0,'SeniorCitizen']='No'
df.loc[df.SeniorCitizen==1,'SeniorCitizen']='Yes'

eda_cat('gender',df)
eda_cat('SeniorCitizen',df)
eda_cat('Partner',df)
eda_cat('Dependents',df)


# In[273]:


### EDA for signed up services


eda_cat('PhoneService',df)
eda_cat('MultipleLines',df)
eda_cat('InternetService',df)
eda_cat('OnlineSecurity',df)
eda_cat('OnlineBackup',df)
eda_cat('DeviceProtection',df)
eda_cat('TechSupport',df)
eda_cat('StreamingTV',df)
eda_cat('StreamingMovies',df)


# In[274]:


### eda for payment features

eda_cat('Contract',df)
eda_cat('PaperlessBilling',df)
eda_cat('PaymentMethod',df)


# In[275]:


### EDA for numerical features


# In[276]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())


# In[277]:


def numeric(feature,df):
    currentFeature = df.groupby([feature, 'Churn']).size().reset_index()
    currentFeature= currentFeature.rename(columns={0: 'Count'})
    fig = px.histogram(currentFeature, x=feature, y='Count', color='Churn', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    fig.show()


# In[278]:


numeric('tenure',df)
numeric('MonthlyCharges',df)
numeric('TotalCharges',df)


# In[279]:


#### some more eda on numerical features for better understanding

numerics=pd.DataFrame()
numerics['tenure']=pd.qcut(df['tenure'], q=3, labels= ['low', 'medium', 'high'])
numerics['MonthlyCharges']=pd.qcut(df['MonthlyCharges'],q=3,labels=['low','medium','high'])
numerics['TotalCharges']=pd.qcut(df['TotalCharges'],q=3,labels=['low','medium','high'])
numerics['Churn']=df['Churn']
eda_cat('tenure',numerics)
eda_cat('MonthlyCharges',numerics)
eda_cat('TotalCharges',numerics)


# # Data PreProcessing for further study

# In[280]:


df.drop(['customerID'],axis=1,inplace=True)


# In[281]:


df['Churn']=df['Churn'].map({'Yes':1,'No':0})
df['PaperlessBilling']=df['PaperlessBilling'].map({'Yes':1,'No':0})
df['PhoneService']=df['PhoneService'].map({'Yes':1,'No':0})
df['SeniorCitizen']=df['SeniorCitizen'].map({'Yes':1,'No':0})
df['Dependents']=df['Dependents'].map({'Yes':1,'No':0})
df['Partner']=df['Partner'].map({'Yes':1,'No':0})
df['gender']=df['gender'].map({'Male':1,'Female':0})
df=pd.get_dummies(df,drop_first=True)


# In[282]:


df


# In[283]:


corr=df.corr()

correlation=px.imshow(corr,width=1000,height=800)
correlation.show()


# In[284]:


corr


# In[285]:


## questions to ponder

## which features will have a higher influence in churn
## which features are paramount to building a high performance model


# In[286]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in df.columns]
df.columns = all_columns


glm=[e for e in all_columns if e not in ['customerID','Churn']]
glm='+'.join(map(str,glm))


glm_model=smf.glm(formula=f'Churn~{glm}',data=df,family=sm.families.Binomial())
res=glm_model.fit()
print(res.summary())


# In[287]:


np.exp(res.params)


# In[288]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df['tenure']=sc.fit_transform(df[['tenure']])
df['MonthlyChanrges']=sc.fit_transform(df[['MonthlyCharges']])
df['TotalCharges']=sc.fit_transform(df[['TotalCharges']])
                                


# In[ ]:





# # Modelling

# In[289]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


##performance metrics

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

## train test split


from sklearn.model_selection import train_test_split
X=df.drop('Churn',axis=1)
y=df['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)


# In[290]:


def prediction(alg,name,params={}):
    model=alg(**params)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    
    
    
    
    def print_scores(alg,y_true,y_predict):
        print(name)
        accuracy=accuracy_score(y_true,y_predict)
        precision=precision_score(y_true,y_predict,zero_division=0)
        recall=recall_score(y_true,y_predict)
        fmeasure=f1_score(y_true,y_predict)
        
        
        print("Accuracy: ",accuracy)
        print("Precision: ",precision)
        print("Recall",recall)
        print("f1 score",f1_score)
    print_scores(alg,y_test,y_pred)
    return model





## logistic regression:

log=prediction(LogisticRegression,'Logistic Regression')
print(log)
print()


# In[291]:


### feature selection

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
log = LogisticRegression()
rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)


X_rfe = X.iloc[:, rfecv.support_]

print("X dimension:",X.shape)
print("X column list:", X.columns.tolist())
print("X_rfe dimension",X_rfe.shape)
print("X_rfe column list:", X_rfe.columns.tolist())


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))







# In[294]:



## SVC:
svc_model = prediction(SVC, 'SVC Classification')
print(svc_model)


## Random Forest
print()
rf_model=prediction(RandomForestClassifier,"Random Forest Classification")
print(rf_model)



## decision tree
print()
dt_model = prediction(DecisionTreeClassifier, "Decision Tree Classification")
print(dt_model)



## Naive bayes
print()
nb_model=prediction(GaussianNB,"Naive Bayes Classification")
print(nb_model)


# In[293]:








### Logistic Regression: hyperparameter tuning:

best_model=LogisticRegression()

from sklearn.model_selection import RepeatedStratifiedKFold
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)


from scipy.stats import loguniform
space=dict()

space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 1000)

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(best_model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)


result = search.fit(X_rfe, y)
params = result.best_params_

log_model = prediction(LogisticRegression, 'Logistic Regression Classification', params=params)


# In[295]:


import joblib
filename='model.sav'
joblib.dump(log, filename)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




