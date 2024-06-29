import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv(r'C:\Users\magal\university_ranking\UniversityRanking_16_17_18_19_20_21.csv')
df=pd.read_csv(r'C:\Users\magal\university_ranking\UniversityRanking_16_17_18_19_20_21.csv')
print(df)

df.describe()

df.info()

x=df.drop(['Institute Id','Institute Name','City','State','Rank','Year'],axis=1)
x=x.values
x

y=df['Rank'].values
y

#splitting the DS into training and testing DS
from sklearn.model_selection import train_test_split
#splitting the DS
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

x_train, x_test, y_train, y_test

import seaborn as sb

sb.pairplot(df)

df.corr()

# Decision Tree Regression model
# importing from the regressor

from sklearn.tree import DecisionTreeRegressor

# Creating and fitting the model

Rank_dt=DecisionTreeRegressor().fit(x_train,y_train)

print('The training r_sq is: %.2f'% Rank_dt.score(x_train,y_train))

# Training model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score 

# Prediction on the DS
ytrain_pred=Rank_dt.predict(x_train)

# The r_sq
print('The r_sq is:%.2f'%r2_score(y_train,ytrain_pred))

# The MAE
print('The MAE is:%.2f'%mean_absolute_error(y_train,ytrain_pred))

# The MSE
print('The MSE is:%.2f'%mean_squared_error(y_train,ytrain_pred))

# The RMSE
import numpy as np
print('The RMSE is:%.2f'%np.sqrt(mean_squared_error(y_train,ytrain_pred)))

# Prediction on the testing data
ytest_pred=Rank_dt.predict(x_test)

# The r_sq
print('The testing r_sq is: %.2f'% r2_score(y_test,ytest_pred))

print('The testing r_sq is: %.2f'% Rank_dt.score(x_test,y_test))

#testing model evaluation

# The MAE
print('The MAE is: %.2f'% mean_absolute_error(y_test,ytest_pred))

# The MSE
print('The MSE is: %.2f'% mean_squared_error(y_test,ytest_pred))

# The RMSE
print('The RMSE is: %.2f'% np.sqrt(mean_squared_error(y_test,ytest_pred)))

#predict new data
new_data=[[82.16,84.54,91.08,75.48,43.7,100]]
Rank_dt.predict(new_data)

#predict new data
new_data=[[37.67,44.8,8.92,60.44,80.33,14.31]]
Rank_dt.predict(new_data)

#predict new data
new_data=[[50,60,56,70,23,60]]
Rank_dt.predict(new_data)

#predict new data
new_data=[[91.81,94.45,96.12,100,67.18,100]]
Rank_dt.predict(new_data)

yr=int(input('Enter the year:'))
colname=input("Enter any one Column Name[Institute Id,Institute Name,City,State,Rank]:")
colname=colname.title()
colvalue=input("Enter corresponding Column Value[Institute Id,Institute Name,City,State,Rank]:")
if colname=='Rank':
    colvalue=int(colvalue)
elif colname=='Institute Id':
    colvalue=colvalue.upper()
else:
    colvalue=colvalue.title()
fetch=data[colname]==colvalue
rec=data[fetch]
record=data[fetch]

record

fet=rec['Year']==yr
rc=rec[fet]
rc

nd=[]
if len(rc)!=0:
    for i,r in rc.iterrows():
        lst=[r.Score,r.TLR,r.RPC,r.GO,r.OI,r.Perception]
        nd.append(lst)
    nd
    print('Rank: %.0f'%Rank_dt.predict(nd))
else:
    print('Data Not Available')