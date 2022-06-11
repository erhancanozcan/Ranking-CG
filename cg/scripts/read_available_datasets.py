#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:08:14 2019

@author: can
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


from scipy.spatial import distance_matrix
from sklearn import tree
import pandas as pd

def selected_data_set(datasetname,location):
    if datasetname=="xor":
        
        location=location+"/xor"
        os.chdir(location)
        data = pd.read_csv('xor_data.csv',sep=',')
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data, class_data, test_size = 0.20, random_state = 5,stratify=class_data)
        df=pd.concat([train_data, train_class], axis=1)
        df_test=pd.concat([test_data,test_class],axis=1)
        
        return df,df_test,test_class,test_data,train_class,train_data
    
    if datasetname=="monks1":
        location=location+"/monks1"
        os.chdir(location)
        
        data=pd.read_csv('monks_1.test.txt',sep=' ')
        data=data.iloc[:,1:]
        data=data.iloc[:,:7]
        data=data.replace('?',np.nan)
        data=data.dropna()
        data['class']=data.a
        data=data.iloc[:,1:]
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        data.loc[data.iloc[:,6]==0, 'class'] = -1
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        
        data=pd.get_dummies(data, columns=["f0","f1","f2","f3","f4","f5"])

        col_names=[]
        col_no=data.shape[1]
        for i in range(col_no):
            col_names.append( "f" + str(i))
        
        data.columns=col_names
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
        
    
    
    elif datasetname=="cleveland_heart":
        
        
        location=location+"/cleveland_heart"
        os.chdir(location)
        data = pd.read_csv('processed.cleveland.data.txt',sep=',')
        data=data.replace('?',np.nan)
        data=data.dropna()
        
        
        
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        data.loc[data.iloc[:,13]==0, 'class'] = -1
        data.loc[data.iloc[:,13]==1, 'class'] = 1
        data.loc[data.iloc[:,13]==2, 'class'] = 1
        data.loc[data.iloc[:,13]==3, 'class'] = 1
        data.loc[data.iloc[:,13]==4, 'class'] = 1
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        data.f11=pd.to_numeric(data.f11)
        data.f12=pd.to_numeric(data.f12)
        data_norm = (data - data.mean()) / (data.std())
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data

    
    
    
    elif datasetname=="parkinsons":
        location=location+"/parkinsons"
        os.chdir(location)
        
        data = pd.read_csv('parkinsons.data.txt',sep=',')
        data=data.drop(['name'], axis=1)
        
        data['class']=data.status
        data=data.drop(['status'], axis=1)
        
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        data.loc[data.iloc[:,22]==0, 'class'] = -1
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        data_norm = (data - data.mean()) / (data.std())
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data

        
    
    
    elif datasetname=="cancer_wbc":
        location=location+"/cancer_wbc"
        os.chdir(location)
        data = pd.read_csv('cancer_wbc.data.txt',sep=',',header=None)
        
        data=data.replace('?',np.nan)
        data=data.dropna()
        data.loc[data.iloc[:,10]==2, 10] = 1
        data.loc[data.iloc[:,10]==4, 10] = -1
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        data=data.drop(['f0'], axis=1)
        data.f6=pd.to_numeric(data.f6)
        
        data_norm=data
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
    
    elif datasetname=="sonar":
        location=location+"/sonar"
        os.chdir(location)
        data = pd.read_csv('sonar_data.txt',sep=',',header=None)

        data=data.replace('?',np.nan)
        data=data.dropna()
        data.loc[data.iloc[:,60]=="R", 60] = 1
        data.loc[data.iloc[:,60]=="M", 60] = -1
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        #data=data.drop(['f0'], axis=1)
        #data.f6=pd.to_numeric(data.f6)
        
        data_norm=data
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
    
    
    elif datasetname=="spectf":
        location=location+"/spectf"
        os.chdir(location)

        data = pd.read_csv('SPECTF.test.txt',sep=',',header=None)
        data_two = pd.read_csv('SPECTF.train.txt',sep=',',header=None)
        data=data.append(data_two)
        
        
        data=data.replace('?',np.nan)
        data=data.dropna()
        data.loc[data.iloc[:,0]==0, 0] = -1
        #data.loc[data.iloc[:,60]=="M", 60] = -1
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        col_names.append("class")
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        #data=data.drop(['f0'], axis=1)
        #data.f6=pd.to_numeric(data.f6)
        
        data_norm=data
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
    


    elif datasetname=="survival_scaled":
        location=location+"/survival"
        os.chdir(location)
        
        data = pd.read_csv('haberman.data.txt',sep=',',header=None)
        data=data.replace('?',np.nan)
        data=data.dropna()
        data.loc[data.iloc[:,3]==2, 3] = -1
        #data.loc[data.iloc[:,60]=="M", 60] = -1
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        #data=data.drop(['f0'], axis=1)
        #data.f6=pd.to_numeric(data.f6)
        
        data_norm = (data - data.mean()) / (data.std())
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
        
    

    
    elif datasetname=="ionosphere":
        location=location+"/ionosphere"
        os.chdir(location)
        data = pd.read_csv('ionosphere.data.txt',sep=',',header=None)
        data=data.drop([1], axis=1)
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        col_names=[]
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        col_names.append("class")
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        data.loc[data['class'] == 'g', 'class'] = 1
        data.loc[data['class'] == 'b', 'class'] = -1
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        
        data_norm = (data - data.mean()) / (data.std())
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
    
    elif datasetname=="votes":
        location=location+"/votes"
        os.chdir(location)
        
        data = pd.read_csv('votes.data.txt',sep=',',header=None)
        data=data.replace('?',np.nan)
        data=data.dropna()
        data=data.replace('y',1)
        data=data.replace('n',0)
        
        
        col_no=data.shape[1]
        row_no=data.shape[0]
        
        
        col_names=[]
        col_names.append("class")
        for i in range(col_no-1):
            col_names.append( "f" + str(i))
        
        
        row_names=[]
        for i in range(row_no):
            row_names.append( "p" + str(i))
        
        data.columns=col_names
        data.index=row_names
        data.loc[data['class'] == 'democrat', 'class'] = 1
        data.loc[data['class'] == 'republican', 'class'] = -1
        
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.3, random_state = 5)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        df_test=pd.concat([test_data,test_class],axis=1)
        del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
    
    else:
        print("Wrong dataset name. Please write one of the followings: xor,monks1,cleveland_heart,parkinsons,cancer_wbc,sonar,spectf,survival_scaled,ionosphere, or votes.")
        return None,None,None,None,None,None
    
    

