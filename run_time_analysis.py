data_location = '/Users/can/Documents/GitHub/Ranking-CG/Datasets'
#data_location = '/home/erhan/Ranking-CG/Datasets'
import os
import sys
import pandas as pd
import numpy as np
from gurobipy import *
from sklearn.tree import DecisionTreeClassifier
#from read_available_datasets import selected_data_set
from sklearn import tree
import matplotlib.pyplot as plt
import scipy
from scipy.stats import iqr
#import cvxpy as cp
#from cvxpy.atoms.pnorm import pnorm
#import dlib
from datetime import date
from scipy.io import arff
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,f1_score
import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse
sys.path.append(os.getcwd())

from cg.scripts.algs.base_srcg import *
from cg.scripts.algs.init_alg import init_alg
from getPerformance import *


class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
          
        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value

nof_features  = {
    'abalone19': 8,
    'yeast6': 8,
    'glass5': 9,
    'ecoli4': 7,
    'vowel0': 13,
    'page-blocks0':10 ,
    'segment0': 19,
    'new-thyroid1':5,
    'vehicle0': 18 ,
    'iris0': 4,
    'wisconsin': 9,
    'poker-9_vs_7':10 ,
    'winequality-red-3_vs_5':11,
    'dermatology-6':34,
    'yeast3':8,
    'vehicle3':18,
    'yeast1':8,
    'vehicle1':18,
    'pima':8,
    'glass0':9,
    'glass-0-1-2-3_vs_4-5-6':9,
    'glass1':9,
    'glass6':9,
    'haberman':3,
    'new-thyroid1':5,
    'new-thyroid2':5,
    'ecoli-0_vs_1':7,
    'ecoli1':7,
    'ecoli2':7,
    'abalone-3_vs_11':8,
    'abalone9-18':8,
    'abalone-21_vs_8':8,
    'cleveland-0_vs_4':13,
    'ecoli-0-1_vs_2-3-5':7,
    'ecoli-0-1_vs_5':7,
    'ecoli-0-1-3-7_vs_2-6':7,
    'ecoli-0-1-4-6_vs_5':7,
    'ecoli-0-1-4-7_vs_2-3-5-6':7,
    'ecoli-0-1-4-7_vs_5-6':7,
    'ecoli-0-2-3-4_vs_5':7,
    'ecoli-0-2-6-7_vs_3-5':7,
    'ecoli-0-3-4_vs_5':7,
    'ecoli-0-3-4-6_vs_5':7,
    'ecoli-0-3-4-7_vs_5-6':7,
    'ecoli-0-4-6_vs_5':7,
    'ecoli-0-6-7_vs_3-5':7,
    'ecoli-0-6-7_vs_5':7,
    'ecoli4':7,
    'flare-F':11,
    'glass-0-1-4-6_vs_2':9,
    'glass-0-1-5_vs_2':9,
    'glass-0-1-6_vs_2':9,
    'glass-0-1-6_vs_5':9,
    'glass-0-4_vs_5':9,
    'glass-0-6_vs_5':9,
    'glass4':9,
    'glass5':9,
    'kddcup-land_vs_portsweep':41,
    'kr-vs-k-zero_vs_eight':6,
    'led7digit-0-2-4-5-6-7-8-9_vs_1':7,
    'lymphography-normal-fibrosis':18,
    'page-blocks-1-3_vs_4':10,
    'poker-8_vs_6':10,
    'poker-8-9_vs_6':10,
    'shuttle-6_vs_2-3':9,
    'shuttle-c2-vs-c4':9,
    'winequality-red-4':11,
    'winequality-red-8_vs_6':11,
    'winequality-red-8_vs_6-7':11,
    'winequality-white-3_vs_7':11,
    'winequality-white-9_vs_4':11,
    'yeast-0-2-5-6_vs_3-7-8-9':8,
    'yeast-0-2-5-7-9_vs_3-6-8':8,
    'yeast-0-3-5-9_vs_7-8':8,
    'yeast-0-5-6-7-9_vs_4':8,
    'yeast-1_vs_7':8,
    'yeast-1-2-8-9_vs_7':8,
    'yeast-1-4-5-8_vs_7':8,
    'yeast-2_vs_4':8,
    'yeast-2_vs_8':8,
    'yeast4':8,
    'yeast5':8,
    'yeast6':8,
    'zoo-3':16
}

if __name__ == '__main__':    
    
    run_start_time=datetime.datetime.now()
    parser = argparse.ArgumentParser()
      
    # adding an arguments 
    parser.add_argument('--kwargs', 
                        nargs='*', 
                        action = keyvalue)
      
     #parsing arguments 
    args = parser.parse_args().kwargs
    
    args={'dname' : 'yeast6'}
    print(args)

    if args['dname'] == 'All':
        data_list = os.listdir(data_location)
    else:
        data_list = [args['dname']]

    all_res = []

    for dname in data_list:
        print(dname)
        try:
            content = os.path.join(data_location,dname, dname+'.dat')
            dt = pd.read_csv(content,  sep=',\s*',
                                engine='python', skiprows=nof_features[dname] + 4, na_values='?').reset_index()

            y = pd.DataFrame((dt['@data']=='positive').values.astype(int)*2-1)
            X_full = dt.drop(columns='@data')
            X_full = pd.get_dummies(X_full,drop_first=True)
        
        except:
            content = os.path.join(data_location,dname, dname+'.data')
            dt = pd.read_csv(content)
            y = dt.iloc[:,-1].to_frame()
            X_full =  dt.iloc[:,:-1]
            X_full = pd.get_dummies(X_full,drop_first=True)
            
            
            
            
            
        feat_perc=[0.2,0.4,0.6,0.8,1.0]
        
        data_perc=[0.2,0.4,0.6,0.8,0.99]
        
        all_res = []
        
        no_f=X_full.shape[1]
        for f_p in feat_perc:
            X=X_full.sample(n=int(no_f*f_p),axis='columns')
            
            
            for d_p in data_perc:
                
                for r_s in range(5):
            
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    stratify=y, 
                                                                    test_size=1-d_p,random_state=r_s)
                    
                   
                    num_pairs=np.sum(y_train.values==1)*np.sum(y_train.values==-1)
                    num_points=y_train.shape[0]
                    num_features=X_train.shape[1]
                    num_points=y_train.shape[0]
                    
                    y_train.columns = ['class']
                    y_test.columns = ['class']
                    dt_train  =X_train.copy()
                    dt_train['class'] = y_train


                    dt_test  =X_test.copy()
                    dt_test['class'] = y_test

                    X_train_distance = scipy.spatial.distance.cdist(X_train,X_train, 'euclidean')
                    X_test_distance = scipy.spatial.distance.cdist(X_test,X_train, 'euclidean')
                    
                    
                    
                    
                    #SVC l2
                    sel_c=0.1
                    start_time = time.time()
                    base_estimator = LinearSVC(penalty = 'l2',C = sel_c,class_weight = None,max_iter=1000000)
                    base_estimator.fit(X_train_distance,y_train.values.reshape(-1))
                    elapsed_time = time.time() - start_time
                    
                    #dataset - method - seed - data_perc - feat_perc - num_points - num_pairs - num_features - opt_time
                    all_res.append([dname, 'SVC_l2']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    #SVC l1
                    sel_c=0.1
                    start_time = time.time()
                    base_estimator = LinearSVC(penalty = 'l1',C = sel_c,class_weight = None,max_iter=1000000,dual=False)
                    base_estimator.fit(X_train_distance,y_train.values.reshape(-1))
                    elapsed_time = time.time() - start_time
                    
                    #dataset - method - seed - data_perc - feat_perc - num_points - num_pairs - num_features - opt_time
                    all_res.append([dname, 'SVC_l1']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    #SVC_l2_weight
                    sel_c=0.1
                    start_time = time.time()
                    base_estimator = LinearSVC(penalty = 'l2',C = sel_c,class_weight = 'balanced',max_iter=1000000)
                    base_estimator.fit(X_train_distance,y_train.values.reshape(-1))
                    elapsed_time = time.time() - start_time
                    
                    #dataset - method - seed - data_perc - feat_perc - num_points - num_pairs - num_features - opt_time
                    all_res.append([dname, 'SVC_l2_weight']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    # SVC_l1_weight
                    sel_c=0.1
                    start_time = time.time()
                    base_estimator = LinearSVC(penalty = 'l1',C = sel_c,class_weight = 'balanced',max_iter=1000000,dual=False)
                    base_estimator.fit(X_train_distance,y_train.values.reshape(-1))
                    elapsed_time = time.time() - start_time
                    
                    #dataset - method - seed - data_perc - feat_perc - num_points - num_pairs - num_features - opt_time
                    all_res.append([dname, 'SVC_l1_weight']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    #ranking_cg
                    
                    # X_train_val = X_train.iloc[train_index]
                    # X_test_val = X_train.iloc[test_index]
                    # y_train_val = y_train.iloc[train_index]
                    # y_test_val = y_train.iloc[test_index]
                    # dt_train_val = dt_train.iloc[train_index,:]
                    # dt_test_val = dt_train.iloc[test_index,:]
                    
                    X_train_val = X_train
                    X_test_val = X_test
                    y_train_val = y_train
                    y_test_val = y_test
                    dt_train_val = dt_train
                    dt_test_val = dt_test
                    
                    
                    #method parameters
                    alg_type='ranking_cg'
                    stp_perc=0#set this approproately
                    stp_cond="tr_roc"
                    lr=1.0
                    alpha=0.1
                    prot_stop_perc=1e-5
                    max_epoch=1000
                    
                    method1=init_alg(alg_type,X_train_val,y_train_val,X_test_val,y_test_val,dt_train_val,dt_test_val,
                                            distance="euclidian",stopping_condition=stp_cond,
                                            stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                                            selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                                            max_epoch=max_epoch)

                    method1.run()
                    
                    all_res.append([dname, 'ranking_cg']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,method1.opt_time])
                    
                    #ranking_cg_prototype
                    alg_type='ranking_cg_prototype'
                    stp_perc=0#set this approproately
                    stp_cond="tr_roc"
                    lr=0.001
                    alpha=0.1
                    prot_stop_perc=1e-5
                    max_epoch=1000
                    
                    method1=init_alg(alg_type,X_train_val,y_train_val,X_test_val,y_test_val,dt_train_val,dt_test_val,
                                            distance="euclidian",stopping_condition=stp_cond,
                                            stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                                            selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                                            max_epoch=max_epoch)

                    method1.run()
                    all_res.append([dname, 'ranking_cg_prototype']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,method1.opt_time])




                    #full_ranking
                    alg_type='full_rank'
                    
                    
                    method1=init_alg(alg_type,X_train,y_train,X_test,y_test,dt_train,dt_test,
                                            distance="euclidian",stopping_condition=stp_cond,
                                            stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                                            selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                                            max_epoch=max_epoch)
                    
                    start_time = time.time()
                    method1.run()
                    
                    elapsed_time = time.time() - start_time
                    
                    all_res.append([dname, 'full_rank']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    
                    #L1 rank
                    
                    alg_type='l1_rank'
                    lr=0 #set this approproately
                    
                    method1=init_alg(alg_type,X_train,y_train,X_test,y_test,dt_train,dt_test,
                                            distance="euclidian",stopping_condition=stp_cond,
                                            stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                                            selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                                            max_epoch=max_epoch)
                    
                    start_time = time.time()
                    method1.run()
                    
                    elapsed_time = time.time() - start_time
                    
                    all_res.append([dname, 'l1_rank']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
                    #L inf rank
                    alg_type='l_inf_rank'
                    lr=0 #set this approproately
                    
                    method1=init_alg(alg_type,X_train,y_train,X_test,y_test,dt_train,dt_test,
                                            distance="euclidian",stopping_condition=stp_cond,
                                            stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                                            selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                                            max_epoch=max_epoch)
                    
                    start_time = time.time()
                    method1.run()
                    
                    elapsed_time = time.time() - start_time
                    
                    all_res.append([dname, 'l_inf_rank']+[r_s,d_p,f_p] + [num_points,num_pairs,num_features,elapsed_time])
                    
                    
        save_date = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
        alg_type='time_all'
            
        save_file='%s_%s_%s.csv'%(dname,alg_type,save_date)
        pd.DataFrame(all_res).to_csv('./logs/'+save_file)
        
        run_end_time=datetime.datetime.now()
        
        print('Time Elapsed: %s'%(run_end_time-run_start_time))
                    




                    
            
            
            