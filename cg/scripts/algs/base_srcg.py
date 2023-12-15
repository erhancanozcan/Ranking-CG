#smooth_ranking_cg
import pandas as pd
import datetime
import math
import numpy as np
from gurobipy import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import random
import itertools
#import tensorflow as tf
#import cvxpy as cp
from scipy import signal
import scipy
from scipy import stats
from scipy.stats import iqr
#from cvxpy.atoms.pnorm import pnorm
from scipy.spatial import distance_matrix
import os
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def calc_pnorm_dist(to_this,from_this,p,dist_type):
    
    col_no=to_this.shape[0]
    row_no=from_this.shape[0]
    
    result=np.zeros(row_no*col_no).reshape(row_no,col_no)
    p=0.5
    for i in range(col_no):
        for j in range(row_no):
            if dist_type=="euclidian":
                result[j,i]=(sum(abs(from_this[j,:]-to_this[i,:])**2))**0.5
            elif dist_type=="sq_euclidian":
                result[j,i]=(sum(abs(from_this[j,:]-to_this[i,:])**2))
            elif dist_type =="pnorm":
                raise NotImplementedError()

    return result


class base_srcg:
    
    def __init__(self,train_data,train_class,
                      test_data,test_class,
                      df,df_test,distance="euclidian",
                      stopping_condition=None,stopping_percentage=0.01,
                      lr=0,
                      selected_col_index=0,scale=True):
        
        
        self.train_data=train_data
        self.train_class=train_class
        self.test_data=test_data
        self.test_class=test_class
        self.df=df
        self.df_test=df_test
        self.distance=distance
        self.stopping_condition=stopping_condition
        self.stopping_percentage=stopping_percentage
        self.lr_init=lr
        self.selected_col_index=selected_col_index
        self.scale=scale
        self.counter=1
        self.set_Q=[]

        
        
    def data_preprocess(self):
        
        class_data=self.df[['class']]
        class_data['counter'] = range(len(self.df))
        df=self.df.drop(['class'], axis=1)
    
        self.pos=class_data.loc[class_data['class']==1]
        self.neg=class_data.loc[class_data['class']==-1]
        self.pos=self.pos.iloc[:,1].values
        self.neg=self.neg.iloc[:,1].values
    
        #import itertools
        pairs = list(itertools.product(self.pos, self.neg))
        data_matrix=df.values
        
        
        if self.distance=="euclidian":
            self.data_distance=pd.DataFrame(distance_matrix(data_matrix, data_matrix), index=df.index, columns=df.index)
            self.full_data_distance=calc_pnorm_dist(data_matrix,data_matrix,-1,"euclidian")
        elif self.distance=="sq_euclidian":
            self.data_distance=pd.DataFrame(distance_matrix(data_matrix, data_matrix), index=df.index, columns=df.index)
            self.data_distance=self.data_distance**2
        else:
            print("not written")
        pairs_distance_dif_table=pd.DataFrame(pairs,columns=['pos_sample','neg_sample'])
        
        
        dimension=(len(pairs_distance_dif_table),df.shape[0])
        pairs_distance_dif_table=pairs_distance_dif_table.values
        self.tmp_dist_city=np.zeros(dimension)
        self.pairs_distance_dif_table=pairs_distance_dif_table#will be used by srcg_prototype
        self.prot_learning=np.zeros((len(pairs_distance_dif_table),df.shape[1]))
        self.data_distance_numpy=self.data_distance.values
        tmp_dim=(len(self.pairs_distance_dif_table),len(data_matrix))
        self.full_tmp_dist=np.zeros(tmp_dim)
        
        self.mean_to_scale_test=np.mean(self.data_distance_numpy,axis=0)
        self.sd_to_scale_test=np.std(self.data_distance_numpy,axis=0)    
        if self.scale==True:    
            self.data_distance_numpy = (self.data_distance_numpy - self.mean_to_scale_test) / (self.sd_to_scale_test)
        
       
        for i in range(len(pairs)):
            #print cntr
            index_pos=pairs_distance_dif_table[i,0]
            index_neg=pairs_distance_dif_table[i,1]
            tmp_dif=self.data_distance_numpy[index_pos,:] - self.data_distance_numpy[index_neg,:]
            self.tmp_dist_city[i,:]=tmp_dif
            self.prot_learning[i,:]=self.train_data.values[index_pos,:]-self.train_data.values[index_neg,:]
            full_tmp_dif=self.full_data_distance[index_pos,:] - self.full_data_distance[index_neg,:]
            self.full_tmp_dist[i,:]=full_tmp_dif
            
        
        #mean_to_scale_test=np.mean(tmp_dist_city,axis=0)
        #sd_to_scale_test=np.std(tmp_dist_city,axis=0)    
            
        #tmp_dist_city = (tmp_dist_city - tmp_dist_city.mean()) / (tmp_dist_city.std())
        
        #gram=np.dot(self.tmp_dist_city.T , self.tmp_dist_city)
        #np.linalg.matrix_rank(self.tmp_dist_city)
        #np.linalg.eig(gram)[0] #to see eigen values
        
        self.training_data_index=df.index.values
        self.col_names=[]
        
        for i in range(self.tmp_dist_city.shape[1]):
            self.col_names.append( "p" + str(i))
        
        
        self.number_of_pairs=len(self.tmp_dist_city)
        self.tmp_dist_city_correlation=sum(self.tmp_dist_city>0)/float(self.tmp_dist_city.shape[0])
        
        
        
    def solve_problem_first_time(self):
        #selected_col_index=0
        self.m = Model("srcg")
        self.m.Params.OutputFlag=0
        
        #used_cols keep the columns used in the model.
        #remained_cols keep the list of columns that can be included to the model.
        
        #first column is selected randomly for now.
        #selected_col_index=0
        
        w_name="w"+self.col_names[self.selected_col_index]
        self.remained_cols=np.copy(self.tmp_dist_city)
        self.remained_cols=np.delete(self.remained_cols,self.selected_col_index,axis = 1) 
        self.remained_cols_name=np.delete(self.col_names,self.selected_col_index)
    
        
        self.used_cols=np.array(self.tmp_dist_city[:,self.selected_col_index],)
        self.used_cols.shape=(self.tmp_dist_city.shape[0],1)
        self.set_Q.append(self.train_data.values[self.selected_col_index,:])
        
        self.used_cols_name=np.array(self.col_names[self.selected_col_index])
        self.used_cols_name=np.append(self.used_cols_name,self.col_names[self.selected_col_index])
        
        #no bounds on weights.
        #self.weights=self.m.addVars(1,lb=-1,ub=1,name=w_name)
        self.weights=self.m.addVars(1,lb=-GRB.INFINITY,name=w_name)
        errors  = self.m.addVars(len(self.pos),len(self.neg),lb=0.0,name="e")
        
        #self.obj=quicksum(errors[f] for f in errors)
        self.obj=quicksum(errors[f] for f in errors)/len(errors)
        self.m.setObjective(self.obj, GRB.MINIMIZE)
        
        #for k in range(len(weights)):
         #   print k
        
        #pair_counter=0
        """for p in errors:
          m.addConstr(quicksum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
          pair_counter=pair_counter+1
        """
        #m.addConstrs(sum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
         
        
        #print ("model constraints are being counstructed")
        #print(datetime.datetime.now())
        
        self.constrain=self.m.addConstrs((errors[i,j] + 
        quicksum(self.weights[k]*(self.tmp_dist_city[i*len(self.neg)+j,self.selected_col_index]) 
        for k in range(len(self.weights))) >= 1 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c')
        
        
        
        
        #print ("end")
        #print(datetime.datetime.now())                    
        
        
        # Solve
        
        
        #0 : primal simplex 1:dual simplex
        self.m.Params.Method=0
        start_time=datetime.datetime.now()		
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_time=(end_time-start_time).seconds
        self.m.getVars()
        
        self.errors_list=np.zeros(1*len(self.pos)*len(self.neg))
        self.errors_list=self.errors_list.reshape(1,len(self.pos),len(self.neg))
        
        for i in range (len(self.pos)):
            for j in range (len(self.neg)):
               self.errors_list[0,i,j]=errors[i,j].X
        
        self.real_training_objective=[np.sum(self.errors_list[0])]
        train_class_numpy=self.train_class.values
        res_train=np.dot(self.data_distance_numpy[:,self.selected_col_index],self.weights[0].X)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
        #accuracy
        self.clf=tree.DecisionTreeClassifier(max_depth=1)
        self.clf.fit(res_with_class.memb.values.reshape(len(res_with_class),1),res_with_class.trainclass.values.reshape(len(res_with_class),1))
        train_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        train_accuracy=accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        #accuracyends    
        
        trainroc=roc_auc_score(res_with_class.trainclass,res_with_class.memb)
        
        trainsense=sensitivity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainspec=specificity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        traingeo=geometric_mean_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainprec=precision_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainfone=f1_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        
        self.duals=np.array(self.m.Pi)
        
        self.train_roc_list=np.array(trainroc)
        self.train_accuracy_list=np.array(train_accuracy)
        self.objective_values=np.array(self.m.objVal)
        
        self.train_sensitivity_list=np.array(trainsense)
        self.train_specificity_list=np.array(trainspec)
        self.train_geometric_mean_list=np.array(traingeo)
        self.train_precision_list=np.array(trainprec)
        self.train_fone_list=np.array(trainfone)
        
        self.weight_record=[]
        
        self.weight_record.append(np.array([self.weights[0].X]))
        
        self.abs_dev_variables=[]
        self.abs_value_pos_constraint_list=[]
        self.abs_value_neg_constraint_list=[]
        
    
    def find_new_column(self):

    
        #a=np.array([[2,3,-2],[-1,0,-3]])
    
        
        self.count_res_dot_product=np.array(self.remained_cols.T*self.duals)
        res_dot_pos_count=np.sum(self.count_res_dot_product>0,axis=1)
        res_dot_neg_count=np.sum(self.count_res_dot_product<0,axis=1)
        res_dot_zero_count=np.sum(self.count_res_dot_product==0,axis=1)
        res_dot_difference_count=abs(res_dot_pos_count-res_dot_neg_count)
        
        
        trimmed_count=abs(stats.trim_mean(self.count_res_dot_product, 0.01,axis=1))
        
        
        
        
        #you take the dot product of dot product of each row with duals.
        start_time=datetime.datetime.now()
        res_dot_product=np.array(abs((self.remained_cols.T).dot(self.duals)))
        self.opt_time+=(datetime.datetime.now()-start_time).seconds
        
        #this part keeps the correlation of features whether + or -
        
        if self.counter==2:
            tmp_deletes=self.used_cols_name[1:]
            del_cntr=0
            for i in tmp_deletes:
                tmp_delete=int(i[1:])
                if del_cntr==0:
                    delete_list=np.array([tmp_delete])
                    del_cntr+=1
                else:
                    delete_list=np.append(delete_list,tmp_delete)
            
            tmp_dist_city_correlation_local=np.delete(self.tmp_dist_city_correlation,delete_list)
            
            indices=np.arange(0,self.tmp_dist_city.shape[1],1)
            #self.correlation_list=list(np.array([indices]))
            self.correlation_list=list(np.array([tmp_dist_city_correlation_local]))
            #self.correlation_list.append(tmp_dist_city_correlation_local)
            self.dot_product_list=list(np.array([res_dot_product]))
        else:
            tmp_deletes=self.used_cols_name[1:]
            del_cntr=0
            for i in tmp_deletes:
                tmp_delete=int(i[1:])
                if del_cntr==0:
                    delete_list=np.array([tmp_delete])
                    del_cntr+=1
                else:
                    delete_list=np.append(delete_list,tmp_delete)
            
            tmp_dist_city_correlation_local=np.delete(self.tmp_dist_city_correlation,delete_list)
            self.correlation_list.append(tmp_dist_city_correlation_local)
            self.dot_product_list.append(res_dot_product)
    
        
        

        index_with_highest=np.argmax(res_dot_product)
        selected_point_name=self.remained_cols_name[index_with_highest]
        self.remained_cols_name=np.delete(self.remained_cols_name,index_with_highest)
        self.selected_col_index=int(selected_point_name[1:])
        
        self.w_name="w"+self.col_names[self.selected_col_index]
        self.set_Q.append(self.train_data.values[self.selected_col_index,:])
        
        tmp=self.tmp_dist_city[:,self.selected_col_index]
        tmp.shape=(self.tmp_dist_city.shape[0],1)
        self.used_cols=np.append(self.used_cols,tmp, axis=1)
        self.used_cols_name=np.append(self.used_cols_name,selected_point_name)
        
        self.remained_cols=np.delete(self.remained_cols,index_with_highest,axis = 1)
        
    
    def predict_test_data(self):
        test_class_numpy=np.array(self.test_class)
        test_data_numpy=self.test_data.values
        #counter=1
        #used_cols_name indexi start from 1.
        tmp=self.used_cols_name[self.counter][:]
        focused_data_point_index=int(tmp[1:])
        focused_data_point_name=self.training_data_index[focused_data_point_index]
        #index_to_scale=int(focused_data_point_name[1:])
        df=self.df.drop(['class'], axis=1)
        focused_data_point=df.loc[focused_data_point_name,]
        focused_data_point=focused_data_point.values
        
        if self.distance=="euclidian":
    
            dist_tmp=(test_data_numpy - focused_data_point)**2
            dist_tmp=dist_tmp.sum(axis=1)
            dist_tmp=np.sqrt(dist_tmp)
        elif self.distance=="sq_euclidian":
    
            dist_tmp=(test_data_numpy - focused_data_point)**2
            dist_tmp=dist_tmp.sum(axis=1)
            #dist_tmp=np.sqrt(dist_tmp)
        
        else:
            print("notavailablethisdistancetype")
        
        
        if self.scale==True:
            dist_tmp = (dist_tmp - self.mean_to_scale_test[focused_data_point_index]) / (self.sd_to_scale_test[focused_data_point_index])
        
        
        if(self.counter==1):
            self.distance_btw_test_and_selected=np.array(dist_tmp)
            self.distance_btw_test_and_selected=self.distance_btw_test_and_selected.reshape(len(self.distance_btw_test_and_selected),1)
        else:
            self.distance_btw_test_and_selected=np.append(self.distance_btw_test_and_selected,dist_tmp.reshape(len(dist_tmp),1),axis=1)
        
        
        np_weights=np.array(self.weights[0].X)
        if (self.counter>1):
            for i in range(self.counter-1):
                np_weights=np.append(np_weights,self.weights[i+1].X)  
        
        
        res=np.dot(self.distance_btw_test_and_selected,np_weights)
        res=res.reshape(len(res),1)
        
        res_with_class=pd.DataFrame({'testclass':test_class_numpy[:,0],'memb':res[:,0]},index=range(len(res)))
        #clf=DecisionTreeClassifier(max_depth=1)
        
        
        testroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
        
        #print res_with_class
        
        test_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        accuracy_percentage=accuracy_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        
        tesense=sensitivity_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tespec=specificity_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tegeo=geometric_mean_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        teprec=precision_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tefone=f1_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        
        if self.counter==1:
            self.test_roc_list=np.array(testroc)
            self.test_accuracy_list=np.array(accuracy_percentage) 
            
            self.test_sensitivity_list=np.array(tesense)
            self.test_specificity_list=np.array(tespec)
            self.test_geometric_mean_list=np.array(tegeo)
            self.test_precision_list=np.array(teprec)
            self.test_fone_list=np.array(tefone)
        else:
            self.test_roc_list=np.append(self.test_roc_list,testroc)
            self.test_accuracy_list=np.append(self.test_accuracy_list,accuracy_percentage)  
            
            self.test_sensitivity_list=np.append(self.test_sensitivity_list,tesense)
            self.test_specificity_list=np.append(self.test_specificity_list,tespec)
            self.test_geometric_mean_list=np.append(self.test_geometric_mean_list,tegeo)
            self.test_precision_list=np.append(self.test_precision_list,teprec)
            self.test_fone_list=np.append(self.test_fone_list,tefone)
        
        
        self.counter=self.counter+1
    
    
    def reference_weights(self):
        raise NotImplementedError
        #return self.weight_record[-1]
    
    def solve_problem_with_new_column(self,lr):
        
        """
        lr: is a parameter that we will use to control the size of gradient steps.
        """
        
        """TODO:
            For different options you need to obtain latest_weights in a different way.
            For example, apply exponential smoothing on the weight_record list. And, get the
            desired latest_weights variable. Currently, we are using a naive strategy.
        """
        
        #latest_weights=self.weight_record[-1]
        latest_weights=self.reference_weights()
        num_variables=self.counter
        if num_variables > 2:
            #remove absolute value related constraints.
            self.m.remove(self.abs_pos)
            self.m.remove(self.abs_neg)
            self.m.update()
        
        new_var=self.selected_col_index
        
        self.weights[len(self.weights)] = self.m.addVar(lb=-GRB.INFINITY,name=self.w_name)
        self.m.update()
        for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
            self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],self.tmp_dist_city[i*len(self.neg)+j,new_var])
        
        
        self.abs_dev_variables.append(self.m.addVar(lb=0.0,name="s"+self.w_name))
        
        
        #obj=self.obj+lr*quicksum(self.abs_dev_variables[i] for i in range(len(self.abs_dev_variables)))
        obj=self.obj+lr*quicksum(var for var in self.abs_dev_variables)
        self.m.setObjective(obj, GRB.MINIMIZE)
        
        
        self.abs_pos=self.m.addConstrs((self.abs_dev_variables[i] - self.weights[i] >= -latest_weights[i] for i in range (num_variables-1)),name="aaa")
        self.abs_neg=self.m.addConstrs((self.abs_dev_variables[i] + self.weights[i] >= +latest_weights[i] for i in range (num_variables-1)),name="bbb")
        #self.abs_value_pos_constraint_list.append(pos)
        
        #m.addConstr(x_pv[0]==self.init_energy,name='c_pv_init_storage')
        
        self.m.update()
        
        self.m.Params.Method=1
        start_time=datetime.datetime.now()
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_time+=(datetime.datetime.now()-start_time).seconds
         
        self.duals=np.array([self.constrain[i,j].Pi for i in range(len(self.pos)) \
                                                    for j in range(len(self.neg))])
        
        tmp_errors=np.zeros(1*len(self.pos)*len(self.neg))
        
        err_counter=0
        record=np.zeros(len(self.pos)*len(self.neg))
        for x in self.m.getVars():
            if x.VarName.find("e")!=-1:
                record[err_counter]=x.X
                err_counter+=1
        record=record.reshape(1,len(self.pos),len(self.neg))
        
        self.errors_list=np.append(self.errors_list,record,axis=0)
        self.real_training_objective.append(np.sum(self.errors_list[-1]))
        
        for i in range(len(self.weights)):
            tmp=self.used_cols_name[i+1]
            
            if i==0:
                self.col_list=np.array(int(tmp[1:]))
                self.weight_list=np.array(self.weights[i].X)
            else:
                self.col_list=np.append(self.col_list,int(tmp[1:]))
                self.weight_list=np.append(self.weight_list,self.weights[i].X)
    
    
        train_class_numpy=self.train_class.values
        res_train=np.dot(self.data_distance_numpy[:,self.col_list],self.weight_list)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
         #accuracy
        self.clf=tree.DecisionTreeClassifier(max_depth=1)
        self.clf.fit(res_with_class.memb.values.reshape(len(res_with_class),1),res_with_class.trainclass.values.reshape(len(res_with_class),1))
        train_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        train_accuracy=accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        #accuracyends
        
        
        trainroc=roc_auc_score(res_with_class.trainclass,res_with_class.memb)
        
        trainsense=sensitivity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainspec=specificity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        traingeo=geometric_mean_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainprec=precision_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainfone=f1_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        

        self.train_roc_list=np.append(self.train_roc_list,trainroc)
        self.train_accuracy_list=np.append(self.train_accuracy_list,train_accuracy)
        self.objective_values=np.append(self.objective_values,self.m.objVal)
        
        self.train_sensitivity_list=np.append(self.train_sensitivity_list,trainsense)
        self.train_specificity_list=np.append(self.train_specificity_list,trainspec)
        self.train_geometric_mean_list=np.append(self.train_geometric_mean_list,traingeo)
        self.train_precision_list=np.append(self.train_precision_list,trainprec)
        self.train_fone_list=np.append(self.train_fone_list,trainfone)
        
        tmp_weight_list=np.array([0])
        for i in range(len(self.weights)):
            tmp_weight_list=np.append(tmp_weight_list,self.weights[i].X)
        tmp_weight_list=tmp_weight_list[1:]
        
        self.weight_record.append(tmp_weight_list)
        
    def schedule_lr(self):
        #self.lr=self.lr
        raise NotImplementedError
    
    def stopping_criteria(self):
        #raise NotImplementedError
        if self.stopping_condition == "tr_obj":
            
            if self.counter>100:
                stopper=False
                return stopper
            
            prev_obj=self.objective_values[len(self.objective_values)-2]
            cur_obj=self.objective_values[len(self.objective_values)-1]
            
            
            stopper=True
            if len(self.train_roc_list)==len(self.train_data):
                stopper=False
            if ((prev_obj-cur_obj)/prev_obj) < self.stopping_percentage:
                stopper=False
            if cur_obj==0:
                stopper=False
            return stopper
        elif self.stopping_condition == "real_tr_obj":
            prev_obj=self.real_training_objective[len(self.real_training_objective)-2]
            cur_obj=self.real_training_objective[len(self.real_training_objective)-1]
            
            if ((prev_obj-cur_obj)/prev_obj) < self.stopping_percentage:
                return False
            if cur_obj==0:
                return False
            return True
        elif self.stopping_condition == "tr_roc":
            
            
            if self.counter>100:
                stopper=False
                return stopper
            
            prev_obj=self.train_roc_list[len(self.train_roc_list)-2]
            cur_obj=self.train_roc_list[len(self.train_roc_list)-1]
            
            stopper=True
            if len(self.train_roc_list)==len(self.train_data):
                stopper=False
            if (abs(prev_obj-cur_obj)/prev_obj) < self.stopping_percentage:
                stopper=False
            if cur_obj==1:
                stopper=False
            return stopper
        
        elif self.stopping_condition == "num_f":
            stopper=True
            if self.stopping_percentage == 1:
                stopper=False
            if len(self.train_roc_list)==self.stopping_percentage:
                stopper=False
            
            return stopper
            

    
    def run(self,plot=False,name=None):
        
        self.lr=self.lr_init
        
        self.data_preprocess()
        self.solve_problem_first_time()
        self.predict_test_data()
        
        stopper=True
        #i=0
        while stopper:
            self.find_new_column()
            self.schedule_lr()
            self.solve_problem_with_new_column(self.lr)
            self.predict_test_data()
            stopper=self.stopping_criteria()
        

            
        
        