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
    
    
    if datasetname=="xor_test":
        
        location=location+"/xor"
        os.chdir(location)
        data = pd.read_csv('xor_data.csv',sep=',')
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data, class_data, test_size = 0.20, random_state = 5,stratify=class_data)
        df=pd.concat([train_data, train_class], axis=1)
        df_test=pd.concat([test_data,test_class],axis=1)
        
        
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])
        
        #test data generation to shade the region.
        test_x=np.arange(-0.4+0.001,+1.6,0.01)
        test_y=np.arange(-0.6+0.001,+1.4,0.01)
        
        #test_x=np.linspace(-3, 3, 120)
        #test_y=np.linspace(0, 11, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
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
    
    elif datasetname=="ellipsoid":
        
        x=np.arange(-8**0.5+0.001,+8**0.5,0.01)

        y_pos=(2-((x**2)/4))**0.5
        y_neg=-(2-((x**2)/4))**0.5


        #x=np.expand_dims(x, 1)


        rng = np.random.default_rng(12345)


        pos=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.expand_dims(pos, 1)


        pos_two=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)


        pos_three=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_three, 1)],axis=1)

        pos_four=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_four, 1)],axis=1)

        pos_five=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_five, 1)],axis=1)

        pos_six=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_six, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        pos=pos[np.arange(len(x)),selected_col]
        pos=np.expand_dims(pos, 1)

        pos=np.concatenate([np.expand_dims(x, 1),pos],axis=1)




        #neg
        neg=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.expand_dims(neg, 1)


        neg_two=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_two, 1)],axis=1)


        neg_three=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_three, 1)],axis=1)

        neg_four=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_four, 1)],axis=1)

        neg_five=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_five, 1)],axis=1)

        neg_six=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_six, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        neg=neg[np.arange(len(x)),selected_col]
        neg=np.expand_dims(neg, 1)

        neg=np.concatenate([np.expand_dims(x, 1),neg],axis=1)

        data=np.concatenate([pos,neg],axis=0)
        
        
        data=pd.DataFrame(data,columns=['f0','f1'])
        data=data.drop(data.sample(frac=.75,random_state=10).index)
        data['class']=1

        #negative data points
        data_neg=np.array([[0,1.1],
                           [-1.9,1.01],
                           [-2.2,-0.9],
                           [-2.5,0.3],
                           [-0.1,-1.2],
                           [1.5,-0.2],
                           [1.75,0.3],
                           [2.2,-0.9],
                           [2.4,0.5],
                           [0.6,1.1],])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=-1

        data = pd.concat([data, data_neg], axis=0)


        #outlier data points
        data_out=np.array([[0,0.9],
                           [-2,0.75],
                           [-1.5,-0.5],
                           [2.5,0.3],
                           [0.1,-1.2]])
        data_out=pd.DataFrame(data_out,columns=['f0','f1'])
        data_out['class']=1

        data = pd.concat([data, data_out], axis=0)
        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-3, 3, 120)
        test_y=np.linspace(-3, 3, 120)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data
        

    elif datasetname=="two_ellipsoid":
        
        x=np.arange(-8**0.5+0.001,+8**0.5,0.01)

        y_pos=(2-((x**2)/4))**0.5
        y_neg=-(2-((x**2)/4))**0.5


        #x=np.expand_dims(x, 1)


        rng = np.random.default_rng(12345)


        pos=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.expand_dims(pos, 1)


        pos_two=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)


        pos_three=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_three, 1)],axis=1)

        pos_four=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_four, 1)],axis=1)

        pos_five=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_five, 1)],axis=1)

        pos_six=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_six, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        pos=pos[np.arange(len(x)),selected_col]
        pos=np.expand_dims(pos, 1)

        pos=np.concatenate([np.expand_dims(x, 1),pos],axis=1)




        #neg
        neg=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.expand_dims(neg, 1)


        neg_two=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_two, 1)],axis=1)


        neg_three=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_three, 1)],axis=1)

        neg_four=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_four, 1)],axis=1)

        neg_five=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_five, 1)],axis=1)

        neg_six=rng.uniform(np.repeat(-8**0.5,len(x)),y_neg,len(x))
        neg=np.concatenate([neg,np.expand_dims(neg_six, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        neg=neg[np.arange(len(x)),selected_col]
        neg=np.expand_dims(neg, 1)

        neg=np.concatenate([np.expand_dims(x, 1),neg],axis=1)

        data=np.concatenate([pos,neg],axis=0)
        #generate second ellipse
        import copy
        shift_data=copy.deepcopy(data)
        shift_data[:,0]=shift_data[:,0]+rng.uniform(7.98,8.02,len(data))

        data=np.concatenate([data,shift_data],axis=0)



        fill_x=np.arange(3,5,0.2)
        fill_y=np.arange(-2.75,2.75,0.2)

        tmp_fill_data=np.array(np.meshgrid(fill_x,fill_y)).T.reshape(-1,2)
        tmp_fill_data=tmp_fill_data+ (rng.uniform(-0.01,+0.01,560)).reshape(280,2)


        data=np.concatenate([data,tmp_fill_data],axis=0)
        #end
        
        data=pd.DataFrame(data,columns=['f0','f1'])
        data=data.drop(data.sample(frac=.75,random_state=10).index)
        data['class']=1

        #negative data points
        data_neg=np.array([[0,1.1],
                           [-1.9,1.01],
                           [-2.2,-0.9],
                           [-2.5,0.3],
                           [-0.1,-1.2],
                           [1.5,-0.2],
                           [1.75,0.3],
                           [2.2,-0.9],
                           [2.4,0.5],
                           [0.6,1.1],
                           [0,1.1],
                            [-1.8+8,1.04],
                            [-2.1+8,-0.94],
                            [-2.3+8,0.31],
                            [-0.1+8,-1.22],
                            [1.5+8,-0.26],
                            [1.95+8,0.33],
                            [2.1+8,-0.94],
                            [2.5+8,0.52],
                            [0.5+8,1.17]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=-1

        data = pd.concat([data, data_neg], axis=0)


        #outlier data points
        data_out=np.array([[0,0.9],
                           [-2,0.75],
                           [-1.5,-0.5],
                           [2.5,0.3],
                           [0.1,-1.2]])
        data_out=pd.DataFrame(data_out,columns=['f0','f1'])
        data_out['class']=1

        data = pd.concat([data, data_out], axis=0)
        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-3, 13, 320)
        test_y=np.linspace(-3, 3, 120)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data        
        

    elif datasetname=="parabol":
        
        x=np.arange(-8**0.5+0.001,+8**0.5,0.01)

        y_pos=((x**2)/1)
        y_neg=((x**2)/1)


        #x=np.expand_dims(x, 1)


        rng = np.random.default_rng(12345)


        #pos=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.expand_dims(pos, 1)

        pos=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.expand_dims(pos, 1)

        #pos_two=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        #pos_three=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_three, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_four=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_four, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_five=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_five, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_six=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_six, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        pos=pos[np.arange(len(x)),selected_col]
        pos=np.expand_dims(pos, 1)

        pos=np.concatenate([np.expand_dims(x, 1),pos],axis=1)
        pos=np.concatenate([pos,np.expand_dims(np.repeat(1,len(pos)), 1)],axis=1)
        
        
        
        
        x=np.arange(-4**0.5+0.001,+4**0.5,0.05)
        y_neg=((x**2)/1)
        
        neg=rng.uniform(y_neg+5,y_neg+6,len(x))
        neg=np.expand_dims(neg, 1)
        neg=np.concatenate([np.expand_dims(x, 1),neg],axis=1)
        neg=np.concatenate([neg,np.expand_dims(np.repeat(-1,len(neg)), 1)],axis=1)
        
        
        data=np.concatenate([pos,neg],axis=0)
        
        #plt.scatter(x=data[:,0],y=data[:,1])

        


        #####here
        data=pd.DataFrame(data,columns=['f0','f1','class'])
        data=data.drop(data.sample(frac=.5,random_state=10).index)
        data.loc[626]['f0']=data.loc[626]['f0'] - 0.1
        data.loc[626]['f1']=data.loc[626]['f1'] + 0.4
        data.loc[626]['class']=1
        
        data.loc[579]['class']=1
        data=data.drop(index=(579))
        
        data.loc[582]['f0']=data.loc[582]['f0'] + 0.1
        data.loc[582]['f1']=data.loc[582]['f1'] + 0.7
        data.loc[582]['class']=1
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])
        #data['class']=1

        #outlier data points
        data_neg=np.array([[0,6],
                           [-1.1,8.01],
                           [2.2,8.0],
                           [-2,10.3],
                           [1.5,9]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=1

        data = pd.concat([data, data_neg], axis=0)
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])

        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-3, 3, 120)
        test_y=np.linspace(0, 11, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data 
    
    

    elif datasetname=="parabol_2":
        
        x=np.arange(-8**0.5+0.001,+8**0.5,0.01)

        y_pos=((x**2)/1)
        y_neg=((x**2)/1)


        #x=np.expand_dims(x, 1)


        rng = np.random.default_rng(12345)


        #pos=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.expand_dims(pos, 1)

        pos=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.expand_dims(pos, 1)

        #pos_two=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        #pos_three=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_three, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_four=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_four, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_five=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_five, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_six=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_six, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        pos=pos[np.arange(len(x)),selected_col]
        pos=np.expand_dims(pos, 1)

        pos=np.concatenate([np.expand_dims(x, 1),pos],axis=1)
        pos=np.concatenate([pos,np.expand_dims(np.repeat(1,len(pos)), 1)],axis=1)
        
        
        
        
        x=np.arange(-4**0.5+0.001,+4**0.5,0.05)
        y_neg=((x**2)/1)
        
        neg=rng.uniform(y_neg+5,y_neg+6,len(x))
        neg=np.expand_dims(neg, 1)
        neg=np.concatenate([np.expand_dims(x, 1),neg],axis=1)
        neg=np.concatenate([neg,np.expand_dims(np.repeat(-1,len(neg)), 1)],axis=1)
        
        
        data=np.concatenate([pos,neg],axis=0)
        
        #plt.scatter(x=data[:,0],y=data[:,1])

        


        #####here
        data=pd.DataFrame(data,columns=['f0','f1','class'])
        data=data.drop(data.sample(frac=.5,random_state=10).index)
        data.loc[626]['f0']=data.loc[626]['f0'] - 0.1
        data.loc[626]['f1']=data.loc[626]['f1'] + 0.4
        data.loc[626]['class']=1
        
        data.loc[579]['class']=1
        data=data.drop(index=(579))
        
        data.loc[582]['f0']=data.loc[582]['f0'] + 0.1
        data.loc[582]['f1']=data.loc[582]['f1'] + 0.7
        data.loc[582]['class']=1
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])
        #data['class']=1

        #outlier data points
        data_neg=np.array([[0,6],
                           [-1.1,8.01],
                           [2.2,8.0],
                           [1.5,9]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=1

        data = pd.concat([data, data_neg], axis=0)
        
        data_neg=np.array([[0,8]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=-1

        data = pd.concat([data, data_neg], axis=0)
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])

        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-3, 3, 120)
        test_y=np.linspace(0, 11, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data 
    
    

    elif datasetname=="parabol_3":
        
        x=np.arange(-8**0.5+0.001,+8**0.5,0.01)

        y_pos=((x**2)/1)
        y_neg=((x**2)/1)


        #x=np.expand_dims(x, 1)


        rng = np.random.default_rng(12345)


        #pos=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.expand_dims(pos, 1)

        pos=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.expand_dims(pos, 1)

        #pos_two=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        #pos_three=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_three, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_four=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_four, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_five=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_five, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)

        #pos_six=rng.uniform(y_pos,np.repeat(8**0.5,len(x)),len(x))
        #pos=np.concatenate([pos,np.expand_dims(pos_six, 1)],axis=1)
        
        pos_two=rng.uniform(y_pos+1,y_pos+3,len(x))
        pos=np.concatenate([pos,np.expand_dims(pos_two, 1)],axis=1)



        selected_col=rng.integers(6,size=len(x))
        pos=pos[np.arange(len(x)),selected_col]
        pos=np.expand_dims(pos, 1)

        pos=np.concatenate([np.expand_dims(x, 1),pos],axis=1)
        pos=np.concatenate([pos,np.expand_dims(np.repeat(1,len(pos)), 1)],axis=1)
        
        
        
        
        x=np.arange(-4**0.5+0.001,+4**0.5,0.05)
        y_neg=((x**2)/1)
        
        neg=rng.uniform(y_neg+5,y_neg+6,len(x))
        neg=np.expand_dims(neg, 1)
        neg=np.concatenate([np.expand_dims(x, 1),neg],axis=1)
        neg=np.concatenate([neg,np.expand_dims(np.repeat(-1,len(neg)), 1)],axis=1)
        
        
        data=np.concatenate([pos,neg],axis=0)
        
        #plt.scatter(x=data[:,0],y=data[:,1])

        


        #####here
        data=pd.DataFrame(data,columns=['f0','f1','class'])
        data=data.drop(data.sample(frac=.5,random_state=10).index)
        data.loc[626]['f0']=data.loc[626]['f0'] - 0.1
        data.loc[626]['f1']=data.loc[626]['f1'] + 0.4
        data.loc[626]['class']=1
        
        data.loc[579]['class']=1
        data=data.drop(index=(579))
        
        data.loc[582]['f0']=data.loc[582]['f0'] + 0.1
        data.loc[582]['f1']=data.loc[582]['f1'] + 0.7
        data.loc[582]['class']=1
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])
        #data['class']=1

        #outlier data points
        data_neg=np.array([[0,6],
                           [-1.1,8.01],
                           [2.2,8.0],
                           [1.5,9]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=1

        data = pd.concat([data, data_neg], axis=0)
        
        #data_neg=np.array([[0,8]])
        #data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        #data_neg['class']=-1

        #data = pd.concat([data, data_neg], axis=0)
        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])

        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-3, 3, 120)
        test_y=np.linspace(0, 12, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data 
    
    
    elif datasetname=="inner_circles":
        
        num_points=100

        np.random.seed(5)

        random_nums=np.random.uniform(0,1,num_points)


        theta = random_nums * 2 * math.pi * 0.85


        posx= 0 + 0.5 * np.cos(theta)+np.random.normal(0,0.03,num_points)
        posy= 0 + 0.5 * np.sin(theta)+np.random.normal(0,0.03,num_points)

        pos=np.concatenate([posx.reshape((-1, 1)),posy.reshape((-1, 1))],axis=1)
        pos=np.concatenate([pos,np.repeat(1,len(random_nums)).reshape(-1,1)],axis=1)
        pos=np.concatenate([pos,np.array([[-0.25,-0.4,-1],
                                          [-0.20,-0.45,-1],
                                          [0.4,-0.1,-1],
                                          [0.4,-0.1,-1],
                                          [0.4,-0.1,-1]])])



        num_points=200
        random_nums=np.random.uniform(0,1,num_points)


        theta = random_nums * 2 * math.pi 
        
        negx= 0 + 1 * np.cos(theta)+np.random.normal(0,0.07,num_points)
        negy= 0 + 1 * np.sin(theta)+np.random.normal(0,0.07,num_points)

        neg=np.concatenate([negx.reshape((-1, 1)),negy.reshape((-1, 1))],axis=1)
        neg=np.concatenate([neg,np.repeat(-1,len(random_nums)).reshape(-1,1)],axis=1)
        neg=np.concatenate([neg,np.array([[-0.25,0.95,1],
                                          [0,0,1]])])


        data=np.concatenate([pos,neg],axis=0)
        
        #plt.scatter(x=data[:,0],y=data[:,1])

        


        #####here
        data=pd.DataFrame(data,columns=['f0','f1','class'])
        data=data.drop(data.sample(frac=.5,random_state=11).index)
        
        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-3+0.001,+3,0.05)
        test_y=np.arange(-3+0.001,+3,0.05)
        
        test_x=np.linspace(-1.5, 1.5, 120)
        test_y=np.linspace(-1.5, 1.5, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data 

        
        


    elif datasetname=="rectangle":
        
        
        
        
        rng = np.random.default_rng(12345)
        
        #x=np.arange(-8**0.5+0.001,+8**0.5,0.01)
        
        x=rng.uniform(-2,2,100)
        y=rng.uniform(0,1,100)
        
        data=np.concatenate([np.expand_dims(x, 1),np.expand_dims(y, 1)],axis=1)
        data=np.concatenate([data,np.expand_dims(np.repeat(1,len(data)), 1)],axis=1)
        
        data=pd.DataFrame(data,columns=['f0','f1','class'])
        
        
        data_neg=np.array([[-1.75,-1.5],
                           [-1.5,-0.8],
                           [-1.25,-1.25],
                           [-1.25,-1.4],
                           [-1.25,-1.6],
                           [-1.1,-1.25],
                           [0,-1.45],
                           [0.75,-1.25],
                           [1,-1],
                           [1.5,-1.2],
                           [1.6,-1.5]])
        data_neg=pd.DataFrame(data_neg,columns=['f0','f1'])
        data_neg['class']=-1

        data = pd.concat([data, data_neg], axis=0)

        #plt.scatter(x=data.f0,y=data.f1,c=data[['class']].values[:,0])
        

        
        row_names=[]
        for i in range(len(data)):
            row_names.append( "p" + str(i))
            
        data.index=row_names
        
        
        class_data=data[['class']]
        data=data.drop(['class'], axis=1)
        
        data_norm=data
        
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_class, test_class = train_test_split(data_norm, class_data, test_size = 0.001, random_state = 2)
        df=pd.concat([train_data, train_class], axis=1)
        #unused_data, test_data, unused_class, test_class = train_test_split(test_data, test_class, test_size = 0.05, random_state = 5)
        
        
        
        #test data generation to shade the region.
        test_x=np.arange(-2+0.001,+2,0.025)
        test_y=np.arange(-1.75+0.001,+1,0.025)
        
        #test_x=np.linspace(-3, 3, 120)
        #test_y=np.linspace(0, 11, 220)
        
        
        test_data=np.array(np.meshgrid(test_x, test_y)).T.reshape(-1,2)
        test_data=pd.DataFrame(test_data,columns=['f0','f1'])
        test_data['class']=1
        test_data.iat[0,2]=-1
        test_data.iat[10,2]=-1
        
        df_test=test_data
        
        test_class=df_test[['class']]
        test_data=df_test.drop(['class'], axis=1)
        
        #df_test=pd.concat([test_data,test_class],axis=1)
        #del class_data,col_names,col_no,data,data_norm,i,row_names
        
        return df,df_test,test_class,test_data,train_class,train_data          
    
    
    else:
        print("Wrong dataset name. Please write one of the followings: xor,monks1,cleveland_heart,parkinsons,cancer_wbc,sonar,spectf,survival_scaled,ionosphere, or votes.")
        return None,None,None,None,None,None
    
    

