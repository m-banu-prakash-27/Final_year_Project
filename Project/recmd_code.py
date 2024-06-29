import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv(r'C:\Users\magal\university_ranking\UniversityRanking_16_17_18_19_20_21.csv')
df=pd.read_csv(r'C:\Users\magal\university_ranking\UniversityRanking_16_17_18_19_20_21.csv')
print(df)

df.describe()

def zscores(df):
    data=df
    df1=df
    cols=list(df1.columns)
    cols.remove('Institute Id')
    cols.remove('Institute Name')
    cols.remove('City')
    cols.remove('State')
    cols.remove('Rank')
    cols.remove('Year')
    for col in cols:
        col_zscore=col+'_ZScore'
        df1[col_zscore] = (df1[col] - df1[col].mean())/df1[col].std(ddof=0)
    df2=df1
    df2.drop('Institute Id',inplace=True,axis=1)
    df2.drop('Institute Name',inplace=True,axis=1)
    df2.drop('City',inplace=True,axis=1)
    df2.drop('State',inplace=True,axis=1)
    df2.drop('Rank',inplace=True,axis=1)
    mvcols=df2.max()
    print(mvcols)
    max_zscore=df1['Score_ZScore'].max()
    max_ztlr=df1['TLR_ZScore'].max()
    max_zrpc=df1['RPC_ZScore'].max()
    max_zgo=df1['GO_ZScore'].max()
    max_zoi=df1['OI_ZScore'].max()
    max_zPrcp=df1['Perception_ZScore'].max()
    print('Maximum Z Scores: ',max_zscore,max_ztlr,max_zrpc,max_zgo,max_zoi,max_zPrcp)
    th_zscore=max_zscore*0.30
    th_ztlr=max_ztlr*0.30
    th_zrpc=max_zrpc*0.30
    th_zgo=max_zgo*0.30
    th_zoi=max_zoi*0.30
    th_zPrcp=max_zPrcp*0.30
    th_zscore,th_ztlr,th_zrpc,th_zgo,th_zoi,th_zPrcp
    df2.drop('Score',inplace=True,axis=1)
    df2.drop('TLR',inplace=True,axis=1)
    df2.drop('RPC',inplace=True,axis=1)
    df2.drop('GO',inplace=True,axis=1)
    df2.drop('OI',inplace=True,axis=1)
    df2.drop('Perception',inplace=True,axis=1)
    df3=df2
    df3['Score Parameter']=df3['Score_ZScore'].ge(th_zscore)
    df3['TLR Parameter']=df3['TLR_ZScore'].ge(th_ztlr)
    df3['RPC Parameter']=df3['RPC_ZScore'].ge(th_zrpc)
    df3['GO Parameter']=df3['GO_ZScore'].ge(th_zgo)
    df3['OI Parameter']=df3['OI_ZScore'].ge(th_zoi)
    df3['Perception Parameter']=df3['Perception_ZScore'].ge(th_zPrcp)
    df3=df3*1
    data['Score Parameter']=df3['Score_ZScore'].ge(th_zscore)
    data['TLR Parameter']=df3['TLR_ZScore'].ge(th_ztlr)
    data['RPC Parameter']=df3['RPC_ZScore'].ge(th_zrpc)
    data['GO Parameter']=df3['GO_ZScore'].ge(th_zgo)
    data['OI Parameter']=df3['OI_ZScore'].ge(th_zoi)
    data['Perception Parameter']=df3['Perception_ZScore'].ge(th_zPrcp)
    data=data*1
    return data

fet16=df['Year']==2016
rc16=df[fet16]
print(rc16)
df16=zscores(rc16)
print(df16)

fet17=df['Year']==2017
rc17=df[fet17]
print(rc17)
df17=zscores(rc17)
print(df17)

fet18=df['Year']==2018
rc18=df[fet18]
print(rc18)
df18=zscores(rc18)
print(df18)

fet19=df['Year']==2019
rc19=df[fet19]
print(rc19)
df19=zscores(rc19)
print(df19)

fet20=df['Year']==2020
rc20=df[fet20]
print(rc20)
df20=zscores(rc20)
print(df20)

fet21=df['Year']==2021
rc21=df[fet21]
print(rc21)
df21=zscores(rc21)
print(df21)

df16

df17

df18

df19

df20

df21

dff=pd.concat([df16,df17,df18,df19,df20,df21])
dff

data

data['Score Parameter']=dff['Score Parameter']
data['TLR Parameter']=dff['TLR Parameter']
data['RPC Parameter']=dff['RPC Parameter']
data['GO Parameter']=dff['GO Parameter']
data['OI Parameter']=dff['OI Parameter']
data['Perception Parameter']=dff['Perception Parameter']

data

#data.to_csv("UniversityRanking_upd.csv",index=False)

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

rec1=rec
rec1.drop('Institute Id',inplace=True,axis=1)
rec1.drop('Institute Name',inplace=True,axis=1)
rec1.drop('City',inplace=True,axis=1)
rec1.drop('State',inplace=True,axis=1)
rec1.drop('Rank',inplace=True,axis=1)
rec1.drop('Score',inplace=True,axis=1)
rec1.drop('TLR',inplace=True,axis=1)
rec1.drop('RPC',inplace=True,axis=1)
rec1.drop('GO',inplace=True,axis=1)
rec1.drop('OI',inplace=True,axis=1)
rec1.drop('Perception',inplace=True,axis=1)
rec

fet=rec['Year']==yr
rc=rec[fet]
rc

if len(rc)!=0:
    dict=rc.to_dict(orient='records')
    print(dict)
else:
    print('Data Not Available')

c=0
if len(rc)!=0:   
    for i in dict:
        print("The college/university needs improvement in: ")
        for j in i:    
            if i[j]==0:
                print(j)
                c+=1
        if c==0:print('Nothing')
else:
    print('Data Not Available')