###############################################################
#importing libraries
###############################################################
import os
#to work with dataframes
import pandas as pd 
#to perform numerical operatiions
import numpy as np  
#to visualise data
import seaborn as sns 
#to partition data
from sklearn.model_selection import train_test_split 
#for importing library for logistic regression
from sklearn.linear_model import LinearRegression 
#importing performance metrics - accuracy score and performance matrix
from sklearn.metrics import accuracy_score,confusion_matrix

import pandas as pd 

os.chdir("/Users/drsmrao/Documents/Pandas/python")
##############################################################
#importing data
##############################################################

data_diabetes=pd.read_csv('diabetes_dataset__2019.csv')

#create deep copy of orginal data
data=data_diabetes.copy(deep=True)

data.describe()

#############################################################
#explanatory data analysis
#1) getting to know the data
#2) Data pre processing
#3) Cross tables, data visualisation etc.
#############################################################

# check variables data types
print(data.info())

### check for missing values
print('Data coulouns with null values:\n',data.isnull().sum())
#(42 missing-pregnancies, 1 missing-Pdiabetes, 1 missing-Diabetic, 4 missing-BMI)

#dealing with missing values
#(pregnancies, diabetic, pdiabetes)-categorial variables, fill null values-class of maximum count(mode) 
data['Pregnancies'].value_counts()
data['Diabetic'].value_counts()
data['Pdiabetes'].value_counts()

#filling missing values
#data=data.dropna(axis=0)
data['Pregnancies'].fillna(data['Pregnancies'].value_counts().index[0],inplace=True)
data['Diabetic'].fillna(data['Diabetic'].value_counts().index[0],inplace=True)
data['Pdiabetes'].fillna(data['Pdiabetes'].value_counts().index[0],inplace=True)

#(BMI-numerical variable), fill with mean/median
data['BMI'].describe()

#replace with mean-low std deviation and similar median
data['BMI'].fillna(data['BMI'].mean(),inplace=True)

#summary of numerical variables
summary_num=data.describe()
print(summary_num)

#summary of catgeorical variables
summary_cat=data.describe(include = 'O')
print(summary_cat)

#reviewing categorical variables for errenous entries
data['Age'].value_counts()
data['Gender'].value_counts()
data['Family_Diabetes'].value_counts()
data['highBP'].value_counts()
data['PhysicallyActive'].value_counts()
data['Smoking'].value_counts()
data['Alcohol'].value_counts()
data['RegularMedicine'].value_counts() #errenous entries
data['JunkFood'].value_counts()
data['Stress'].value_counts()
data['BPLevel'].value_counts() #errounous entries
data['Pdiabetes'].value_counts() #erronous entries
data['UrinationFreq'].value_counts()
data['Diabetic'].value_counts() #errenous entries

print(np.unique(data['weight']))
print(np.unique(data['Time']))
print(np.unique(data['Chick']))
print(np.unique(data['Diet']))

#replacing erronous entries
data['RegularMedicine'].replace('o','no',inplace=True)
data['BPLevel'].replace('High','high',inplace=True)
data['BPLevel'].replace('Low','low',inplace=True)
data['BPLevel'].replace('normal ','normal',inplace=True)
data['Pdiabetes'].replace('0','no',inplace=True)
data['Diabetic'].replace(' no','no',inplace=True)

#converting object to categorical variables
#data['Age']=data['Age'].astype('category')
#data['Gender']=data['Gender'].astype('category')
#data['Fammily_Diabetes']=data['Family_Diabetes'].astype('category')
#data['highBP']=data['highBP'].astype('category')
#data['PhysicallyActive']=data['PhysicallyActive'].astype('category')
#data['Smoking']=data['Smoking'].astype('category')
#data['Alcohol']=data['Alcohol'].astype('category')
#data['RegularMedicine']=data['RegularMedicine'].astype('category')
#data['JunkFood']=data['JunkFood'].astype('category')
#data['Stress']=data['Stress'].astype('category')
#data['BPLevel']=data['BPLevel'].astype('category')
#data['Pdiabetes']=data['Pdiabetes'].astype('category')
#data['UrinationFreq']=data['UrinationFreq'].astype('category')
#data['Diabetic']=data['Diabetic'].astype('object')

#Checking for multicollinearity between variables
correlation=data.corr()
print(data.corr())

data.columns
#Diabetic
sns.countplot(data['Diabetic'])
diabetic_prop=pd.crosstab(index=data['Diabetic'],columns='count',normalize=True)
print(diabetic_prop)
data['Diabetic'].value_counts()
#gender
sns.countplot(data['Gender'])
gender_prop=pd.crosstab(index=data['Gender'],columns='count',normalize=True)
print(gender_prop)
data['Gender'].value_counts()
#Gender vs Diabetic
sns.countplot(y=data['Gender'],hue='Diabetic',data=data)
gender_diab=pd.crosstab(index=data['Gender'],columns=data['Diabetic'],margins=True,normalize='index')
print(gender_diab)

#age
sns.countplot(data['Age'])
age_prop=pd.crosstab(index=data['Age'],columns='count',normalize=True)
print(age_prop) 
#Age vs Diabetic
sns.countplot(y=data['Age'],hue='Diabetic',data=data)
age_diab=pd.crosstab(index=data['Age'],columns=data['Diabetic'],margins=True,normalize='index')
print(age_diab)


#Family Diabetes
sns.countplot(data['Family_Diabetes'])
famdib_prop=pd.crosstab(index=data['Family_Diabetes'],columns='count',normalize=True)
print(famdib_prop)
#Family diabetes vs Diabetic
sns.countplot(y=data['Family_Diabetes'],hue='Diabetic',data=data)
famdiab_diab=pd.crosstab(index=data['Family_Diabetes'],columns=data['Diabetic'],margins=True,normalize='index')
print(famdiab_diab)

#highBP
sns.countplot(data['highBP'])
highbp_prop=pd.crosstab(index=data['highBP'],columns='count',normalize=True)
print(highbp_prop)
#highBP vs Diabetic
sns.countplot(y=data['highBP'],hue='Diabetic',data=data)
highbp_diab=pd.crosstab(index=data['highBP'],columns=data['Diabetic'],margins=True,normalize='index')
print(highbp_diab)

#Physically Active
sns.countplot(data['PhysicallyActive'])
phyact_prop=pd.crosstab(index=data['PhysicallyActive'],columns='count',normalize=True)
print(phyact_prop)
#Physically active vs Diabetic
sns.countplot(y=data['PhysicallyActive'],hue='Diabetic',data=data)
phyact_diab=pd.crosstab(index=data['PhysicallyActive'],columns=data['Diabetic'],margins=True,normalize='index')
print(phyact_diab)

#BMI
sns.distplot(data['BMI'],bins=10,kde=False)
bmi_prop=pd.crosstab(index=data['BMI'],columns='count',normalize=True)
print(bmi_prop)
#BMI vs Diabetic
sns.boxplot('Diabetic','BMI',data=data)
data.groupby('Diabetic')['BMI'].median()

#Smoking   - drop
sns.countplot(data['Smoking'])
smoke_prop=pd.crosstab(index=data['Smoking'],columns='count',normalize=True)
print(smoke_prop)
#Family diabetes vs Diabetic
sns.countplot(y=data['Smoking'],hue='Diabetic',data=data)
smoke_diab=pd.crosstab(index=data['Smoking'],columns=data['Diabetic'],margins=True,normalize='index')
print(smoke_diab)

#Alcohol
sns.countplot(data['Alcohol'])
alcohol_prop=pd.crosstab(index=data['Alcohol'],columns='count',normalize=True)
print(alcohol_prop)
#Alcohol vs Diabetic
sns.countplot(y=data['Alcohol'],hue='Diabetic',data=data)
alcohol_diab=pd.crosstab(index=data['Alcohol'],columns=data['Diabetic'],margins=True,normalize='index')
print(alcohol_diab)

#Sleep
sns.countplot(data['Sleep'])
sleep_prop=pd.crosstab(index=data['Sleep'],columns='count',normalize=True)
print(sleep_prop)
#Sleep vs Diabetic
sns.countplot(y=data['Sleep'],hue='Diabetic',data=data)
sleep_diab=pd.crosstab(index=data['Sleep'],columns=data['Diabetic'],margins=True,normalize='index')
print(sleep_diab)

#Sound sleep    - drop
sns.countplot(data['SoundSleep'])
ssleep_prop=pd.crosstab(index=data['SoundSleep'],columns='count',normalize=True)
print(ssleep_prop)
#Sound sleep vs Diabetic
sns.countplot(y=data['SoundSleep'],hue='Diabetic',data=data)
ssleep_diab=pd.crosstab(index=data['SoundSleep'],columns=data['Diabetic'],margins=True,normalize='index')
print(ssleep_diab)

#Regular Medicine
sns.countplot(data['RegularMedicine'])
regmed_prop=pd.crosstab(index=data['RegularMedicine'],columns='count',normalize=True)
print(regmed_prop)
#Regular Medicine vs Diabetic
sns.countplot(y=data['RegularMedicine'],hue='Diabetic',data=data)
regmed_diab=pd.crosstab(index=data['RegularMedicine'],columns=data['Diabetic'],margins=True,normalize='index')
print(regmed_diab)

#Junk Food   - drop
sns.countplot(data['JunkFood'])
junkfood_prop=pd.crosstab(index=data['JunkFood'],columns='count',normalize=True)
print(junkfood_prop)
#Junk Food vs Diabetic
sns.countplot(y=data['JunkFood'],hue='Diabetic',data=data)
junkfood_diab=pd.crosstab(index=data['JunkFood'],columns=data['Diabetic'],margins=True,normalize='index')
print(junkfood_diab)

#Stress
sns.countplot(data['Stress'])
stress_prop=pd.crosstab(index=data['Stress'],columns='count',normalize=True)
print(stress_prop)
#Stress vs Diabetic
sns.countplot(y=data['Stress'],hue='Diabetic',data=data)
stress_diab=pd.crosstab(index=data['Stress'],columns=data['Diabetic'],margins=True,normalize='index')
print(stress_diab)

#BPLevel
sns.countplot(data['BPLevel'])
bp_prop=pd.crosstab(index=data['BPLevel'],columns='count',normalize=True)
print(bp_prop)
#BP vs Diabetic
sns.countplot(y=data['BPLevel'],hue='Diabetic',data=data)
bp_diab=pd.crosstab(index=data['BPLevel'],columns=data['Diabetic'],margins=True,normalize='index')
print(bp_diab)

#Pregnancies
sns.distplot(data['Pregnancies'],bins=10,kde=False)
preg_prop=pd.crosstab(index=data['Pregnancies'],columns='count',normalize=True)
print(preg_prop)
#Pregnancies vs Diabetic
sns.countplot(y=data['Pregnancies'],hue='Diabetic',data=data)
preg_diab=pd.crosstab(index=data['Pregnancies'],columns=data['Diabetic'],margins=True,normalize='index')
print(preg_diab)

#Pdiabetes
sns.countplot(data['Pdiabetes'])
pdiab_prop=pd.crosstab(index=data['Pdiabetes'],columns='count',normalize=True)
print(pdiab_prop)
#Pdiabetes vs Diabetic
sns.countplot(y=data['Pdiabetes'],hue='Diabetic',data=data)
pdiab_diab=pd.crosstab(index=data['Pdiabetes'],columns=data['Diabetic'],margins=True,normalize='index')
print(pdiab_diab)

#Urination Frequency
sns.countplot(data['UrinationFreq'])
urifreq_prop=pd.crosstab(index=data['UrinationFreq'],columns='count',normalize=True)
print(urifreq_prop)
#Urination Frequency vs Diabetic
sns.countplot(y=data['UrinationFreq'],hue='Diabetic',data=data)
urifreq_diab=pd.crosstab(index=data['UrinationFreq'],columns=data['Diabetic'],margins=True,normalize='index')
print(urifreq_diab)

#Dropping columns, -Junk Food, SoundSleep, Smoking 
data.drop('JunkFood',axis=1,inplace=True)
data.drop('SoundSleep',axis=1,inplace=True)
data.drop('Smoking',axis=1,inplace=True)
#data.info()

#Reindexing Diabetic names to 0,1
data['Diabetic']=data['Diabetic'].map({'yes':0,'no':1})

final_data=pd.get_dummies(data,drop_first=True)

#storing column names
columns_list=list(final_data.columns)
print(columns_list)

#storing the input/independent variable in features
features=list(set(columns_list)-set(['Diabetic']))
print(features)

#storing dependent/output variable values in y
y=final_data['Diabetic'].values
print(y)

#storing independent/input variable values in x
x=final_data[features].values
print(x)

#splitting data to train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)

###############################################################################
#Logistic Regression
###############################################################################
from sklearn.model_selection import GridSearchCV
#Instance of logistic regression
#logistic=LogisticRegression()
logistic=GridSearchCV(LogisticRegression(), {'C':np.linspace(start=1, stop=100)})
#fitting values for x and y
logistic.fit(train_x,train_y)

#logistic.coef_
#logistic.intercept_

#Prediction from test data
prediction_LR= logistic.predict(test_x)
#print(prediction_LR)

#confusion matrix
confusion_matrix_LR=confusion_matrix(test_y,prediction_LR)
print(confusion_matrix_LR)

#printing miscalssified values
print('Misclassified samples in logistic regression model: %d' %(test_y!=prediction_LR).sum())

#calculating accuracy
accuracy_score_LR=accuracy_score(test_y,prediction_LR)
print(accuracy_score_LR)
logistic.best_params_
logistic.best_score_

###############################################################################
#KNN
###############################################################################

#importing library of knn
from sklearn.neighbors import KNeighborsClassifier

#Storing the K nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=4)

#fitting values for x and y
KNN_classifier.fit(train_x,train_y)

#predicting using model
prediction_KNN=KNN_classifier.predict(test_x)

#confusion matrix
confusion_matrix_KNN=confusion_matrix(test_y,prediction_KNN)
print(confusion_matrix_KNN)

#printing miscalssified values
print('Misclassified samples in KNN model: %d' %(test_y!=prediction_KNN).sum())

#checking accuracy score
accuracy_score_KNN=accuracy_score(test_y,prediction_KNN)
print(accuracy_score_KNN)

#calculating error for k value between 1 and 20
for i in range(1, 20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    accuracy_score_KNN_i=accuracy_score(test_y,pred_i)
    print('Accuracy score for n=',i,'is ',accuracy_score_KNN_i)
    print('Misclassified samples for n=',i,'is: %d'%(test_y!=pred_i).sum())  

   
###############################################################################
#Random Forest    
###############################################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

n_estimators=[int(x) for x in np.linspace(start=1, stop=100)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(start=10, stop=110)]
max_depth.append(None)
min_samples_split=[2,5,10]
min_samples_leaf=[1,2,4]
bootstrap=[True,False]

clf=GridSearchCV(RandomForestRegressor(), {'n_estimators':n_estimators})

#rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf= 3, random_state=1)
#model_rf=rf.fit(train_x,train_y)
model_rf=clf.fit(train_x,train_y)

r2_rf_test=model_rf.score(test_x,test_y)
r2_rf_train=model_rf.score(train_x,train_y)
print(r2_rf_test,r2_rf_train)
model_rf.best_params_



