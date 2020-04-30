#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#from collections import Counter
import pickle

dataset=pd.read_excel("daily data fillna 02042020.xlsx")
dataset.columns
dataset.isna().sum()
dataset1=dataset.loc[:,["SF_speed","BM","Cement_CaO","Cl_C3A","Cl_C3s","GA_Dosage","C_Blaine","RP_Power","PA_prop","Gypsum_Purity","C_Residue","Cement_So3","C1DStrength"]]

plt.hist(dataset1["SF_speed"])

dataset1.skew()
dataset1['Cl_C3A_x']=1/(max(dataset1['Cl_C3A']+1)-dataset1['Cl_C3A'])

dataset1.columns
dataset1.skew()
max(dataset1['Cl_C3A'])
dataset1['GA_Dosage_x']=np.where(dataset1['GA_Dosage']==0,1400,dataset1['GA_Dosage'])
plt.hist(dataset1['GA_Dosage_x'])
dataset1.skew()
dataset1['Cement_So3_x']=np.where(dataset1['Cement_So3']>3,2.558,dataset1['Cement_So3'])
dataset1.skew()
dataset2=dataset1.loc[:,["BM","Cement_CaO","Cl_C3A_x","Cl_C3s","GA_Dosage_x","C_Blaine","RP_Power","PA_prop","Gypsum_Purity","Cement_So3_x","C1DStrength"]]
def stand(x):
    return((x-np.mean(x))/np.std(x))

np.mean(dataset1['Cement_So3'])
dataset3=dataset2.apply(lambda x: stand(x))
dataset3.describe()
mod1=DBSCAN(eps=3,min_samples=4).fit(dataset3)


# DBSCAN model with parameters
model = DBSCAN(eps=3, min_samples=4).fit(dataset3)

# Creating Panda DataFrame with Labels for Outlier Detection
outlier_df = pd.DataFrame(dataset3)

# Printing total number of values for each label
#print(Counter(mod1.labels_))


# Printing DataFrame being considered as Outliers -1
print(outlier_df[mod1.labels_ == -1])

# Printing and Indicating which type of object outlier_df is
print(type(outlier_df))

x1=np.mean(dataset3)
x2=np.std(dataset3) 
dataset4=dataset3*dataset2.std()+dataset2.mean()
round(dataset4.head(),2)
dataset2.head()
dataset4=dataset4[mod1.labels_ != -1]

X = dataset4.iloc[:, 0:10].values
y = dataset4.iloc[:, 10].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, max_features=3, random_state = 123) 
 
# fit the regressor with x and y data 
regressor.fit(X_train, y_train) 
importance=list(np.array(regressor.feature_importances_))
importance
dataset4.columns
dataset["Cl_C3A"].max()+1
x2=5.6772
x2
X_test[0]
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,4,6,7,7,21,34,3]]))
