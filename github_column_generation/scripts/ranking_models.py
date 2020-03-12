#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:33:16 2019

@author: can
"""
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
import tensorflow as tf
import cvxpy as cp
from scipy import signal
import scipy
from scipy import stats
from scipy.stats import iqr
from cvxpy.atoms.pnorm import pnorm
from scipy.spatial import distance_matrix
import os

def calc_pnorm_dist(to_this,from_this,p,dist_type):
    
    col_no=to_this.shape[0]
    row_no=from_this.shape[0]
    
    result=np.zeros(row_no*col_no).reshape(row_no,col_no)
    p=0.5
    for i in range(col_no):
        for j in range(row_no):
            if dist_type=="euclidian":
                result[j,i]=(sum(abs(from_this[j,:]-to_this[i,:])**2))**0.5
            elif dist_type =="pnorm":
                tmp_val=pnorm(abs(from_this[j,:]-to_this[i,:]),p)
                result[j,i]=tmp_val.value

    return result

class rankSVM_column_generation:
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,distance="euclidian",stopping_condition="obj",stopping_percentage=0.01,selected_col_index=0,scale=True,ignore_initial=False,manual_selection=False,manuel_select_counter=0,epsilon_int=0.001):
        self.train_data=train_data
        self.train_class=train_class
        self.test_data=test_data
        self.test_class=test_class
        self.df=df
        self.df_test=df_test
        self.distance=distance
        self.stopping_condition=stopping_condition
        self.stopping_percentage=stopping_percentage
        self.selected_col_index=selected_col_index
        self.scale=scale
        self.counter=1
        self.ignore_initial=ignore_initial
        self.manual_selection=manual_selection
        self.manuel_select_counter=manuel_select_counter
        self.epsilon_int=epsilon_int
    def data_preprocess(self):
        
        class_data=self.df[['class']]
        class_data['counter'] = range(len(self.df))
        df=self.df.drop(['class'], axis=1)
    
        self.pos=class_data.loc[class_data['class']==1]
        self.neg=class_data.loc[class_data['class']==-1]
        self.pos=self.pos.iloc[:,1].values
        self.neg=self.neg.iloc[:,1].values
    
        import itertools
        pairs = list(itertools.product(self.pos, self.neg))
        data_matrix=df.values
        
        
        if self.distance=="euclidian":
            self.data_distance=pd.DataFrame(distance_matrix(data_matrix, data_matrix), index=df.index, columns=df.index)
        else:
            print("not written")
        pairs_distance_dif_table=pd.DataFrame(pairs,columns=['pos_sample','neg_sample'])
        
        
        dimension=(len(pairs_distance_dif_table),df.shape[0])
        pairs_distance_dif_table=pairs_distance_dif_table.values
        self.tmp_dist_city=np.zeros(dimension)
        self.data_distance_numpy=self.data_distance.values
        
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
            
        
        #mean_to_scale_test=np.mean(tmp_dist_city,axis=0)
        #sd_to_scale_test=np.std(tmp_dist_city,axis=0)    
            
        #tmp_dist_city = (tmp_dist_city - tmp_dist_city.mean()) / (tmp_dist_city.std())
        
        self.training_data_index=df.index.values
        self.col_names=[]
        
        for i in range(self.tmp_dist_city.shape[1]):
            self.col_names.append( "p" + str(i))
        
        
        self.number_of_pairs=len(self.tmp_dist_city)
        self.tmp_dist_city_correlation=sum(self.tmp_dist_city>0)/float(self.tmp_dist_city.shape[0])
    
    def solve_problem_first_time(self):
        #selected_col_index=0
        self.m = Model("small_v3")
        
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
        
        self.used_cols_name=np.array(self.col_names[self.selected_col_index])
        self.used_cols_name=np.append(self.used_cols_name,self.col_names[self.selected_col_index])
        
        self.weights=self.m.addVars(1,lb=-1,ub=1,name=w_name)
        #self.weights=self.m.addVars(1,lb=-GRB.INFINITY,name=w_name)
        errors  = self.m.addVars(len(self.pos),len(self.neg),lb=0.0,name="e")
        
        self.m.setObjective(sum(errors[f] for f in errors), GRB.MINIMIZE)
        
        #for k in range(len(weights)):
         #   print k
        
        #pair_counter=0
        """for p in errors:
          m.addConstr(quicksum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
          pair_counter=pair_counter+1
        """
        #m.addConstrs(sum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
         
        
        print ("model constraints are being counstructed")
        print(datetime.datetime.now())
        
        self.constrain=self.m.addConstrs((errors[i,j] + 
        quicksum(self.weights[k]*(self.tmp_dist_city[i*len(self.neg)+j,self.selected_col_index]) 
        for k in range(len(self.weights))) >= 1 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c')
        
        
        
        
        print ("end")
        print(datetime.datetime.now())                    
        
        
        # Solve
        
        
        #0 : primal simplex 1:dual simplex
        self.m.Params.Method=0
        start_time=datetime.datetime.now()		
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_difference=(end_time-start_time).seconds
        self.m.getVars()
        
        self.errors_list=np.zeros(1*len(self.pos)*len(self.neg))
        self.errors_list=self.errors_list.reshape(1,len(self.pos),len(self.neg))
        
        for i in range (len(self.pos)):
            for j in range (len(self.neg)):
               self.errors_list[0,i,j]=errors[i,j].X
        
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
        
        self.duals=np.array(self.m.Pi)
        
        self.train_roc_list=np.array(trainroc)
        self.train_accuracy_list=np.array(train_accuracy)
        self.objective_values=np.array(self.m.objVal)
        
        self.weight_record=list(np.array([[0]]))
        
        self.weight_record.append(np.array([self.weights[0].X]))
        
        
    
    def is_pareto_efficient_simple(self,costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient
        
        
        
    def find_new_column(self):

        
        #a=np.array([[2,3,-2],[-1,0,-3]])

        
        self.count_res_dot_product=np.array(self.remained_cols.T*self.duals)
        res_dot_pos_count=np.sum(self.count_res_dot_product>0,axis=1)
        res_dot_neg_count=np.sum(self.count_res_dot_product<0,axis=1)
        res_dot_zero_count=np.sum(self.count_res_dot_product==0,axis=1)
        res_dot_difference_count=abs(res_dot_pos_count-res_dot_neg_count)
        
        
        trimmed_count=abs(stats.trim_mean(self.count_res_dot_product, 0.01,axis=1))
        
        
        
        
        #you take the dot product of dot product of each row with duals.
        res_dot_product=np.array(abs((self.remained_cols.T).dot(self.duals)))
        
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

        
        
        if self.manual_selection == True:
            #f, ax = plt.subplots(1)
            #ax.plot(self.dot_product_list[len(self.dot_product_list)-1],self.correlation_list[len(self.correlation_list)-1],"bo")
            #ax.set_xlabel("dot_product")
            #ax.set_ylabel("correlation")
            #f.suptitle('Dot_product  vs correlation iter=' + str(len(self.dot_product_list)), fontsize=12)
            #f
            #plt.plot(self.dot_product_list[0],self.correlation_list[0],'bo')
            
            #raw_input("Press enter to see dual*difference count...")
            print("pos_count",res_dot_pos_count)
            print("neg_count",res_dot_neg_count)            
            print("zero_count",res_dot_zero_count)
            print("abs_dif_count",res_dot_difference_count)
            print("index_w/ highest difference:  " , np.argmax(res_dot_difference_count))
            print("index_w/ highest trimmed_mean:  " , np.argmax(trimmed_count))
            
            
            #raw_input("Press enter to see summary(sorted by dotproduct)...")
            iteration_summary=pd.DataFrame({'Dot_product':self.dot_product_list[len(self.dot_product_list)-1],'correlation':self.correlation_list[len(self.correlation_list)-1] - 0.5})
            #pareto optimal
            iteration_summary_pareto=pd.DataFrame({'Dot_product':-1*self.dot_product_list[len(self.dot_product_list)-1],'correlation':-1*abs(self.correlation_list[len(self.correlation_list)-1] - 0.5)})
            
            pareto_summary=self.is_pareto_efficient_simple(iteration_summary_pareto.values)
            print(iteration_summary_pareto)
            #raw_input("press enter to see pareto efficients...")
            print(pareto_summary)
            #raw_input("press enter to continue...")
            pareto_efficient_list=np.where(pareto_summary)[0]
            print(pareto_efficient_list)
            #raw_input("press enter to continue...")
            
            random_select=random.randint(0,len(pareto_efficient_list)-1)
            random_select=pareto_efficient_list[random_select]
            print(random_select)
            #raw_input("press enter to continue...")
            
            #pareto optimal end
            
            
            
            iteration_summary=iteration_summary.sort_values(by='Dot_product',ascending=False)
            print(iteration_summary) 
            
            #raw_input("Press enter to see summary(sorted by correlation)...")
            iteration_summary=iteration_summary.sort_values(by='correlation',ascending=True)
            print(iteration_summary) 
            
            #if you uncomment below line, code waits you to enter an index.
            #index_with_highest=input("enter the index")
            
            #to select random pareto optimal use the line below.
            index_with_highest=random_select
            
            """
            #manuel_select_counter used to plot 2D difference plots and weights. it is useless.
            #index_with_highest=self.manuel_select_counter
            #self.cur_res_dot_product=res_dot_product[index_with_highest]
            ###
            """
            #select index by multiplying correlation and dot product.
            multiplication=np.argmax(abs(iteration_summary.values[:,0]*iteration_summary.values[:,1]))
            #index_with_highest=multiplication
            
            
            
            selected_point_name=self.remained_cols_name[index_with_highest]
            self.remained_cols_name=np.delete(self.remained_cols_name,index_with_highest)
            self.selected_col_index=int(selected_point_name[1:])
            
        else:
            index_with_highest=np.argmax(res_dot_product)
            selected_point_name=self.remained_cols_name[index_with_highest]
            self.remained_cols_name=np.delete(self.remained_cols_name,index_with_highest)
            self.selected_col_index=int(selected_point_name[1:])
        
        self.w_name="w"+self.col_names[self.selected_col_index]
        
        tmp=self.tmp_dist_city[:,self.selected_col_index]
        tmp.shape=(self.tmp_dist_city.shape[0],1)
        self.used_cols=np.append(self.used_cols,tmp, axis=1)
        self.used_cols_name=np.append(self.used_cols_name,selected_point_name)
        
        self.remained_cols=np.delete(self.remained_cols,index_with_highest,axis = 1)

    def solve_problem_with_new_column(self):
       
        
        new_var=self.selected_col_index
        
        #self.weights[len(self.weights)] = self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,name=self.w_name)
        self.weights[len(self.weights)] = self.m.addVar(lb=-1, ub=1,name=self.w_name)
        for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
            self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],self.tmp_dist_city[i*len(self.neg)+j,new_var])
    
        
        self.m.update()
    
    
    

        
        
        self.m.Params.Method=1
        start_time=datetime.datetime.now()
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_difference=self.opt_difference+(end_time-start_time).seconds
        self.m.getVars()
        self.duals=np.array(self.m.Pi)
        
        
        
        
        tmp_errors=np.zeros(1*len(self.pos)*len(self.neg))
        
        err_counter=0
        record=np.zeros(len(self.pos)*len(self.neg))
        for x in self.m.getVars():
            if x.VarName.find("e")!=-1:
                record[err_counter]=x.X
                err_counter+=1
        record=record.reshape(1,len(self.pos),len(self.neg))
        
        self.errors_list=np.append(self.errors_list,record,axis=0)
        
        
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
        

        self.train_roc_list=np.append(self.train_roc_list,trainroc)
        self.train_accuracy_list=np.append(self.train_accuracy_list,train_accuracy)
        self.objective_values=np.append(self.objective_values,self.m.objVal)
        
        tmp_weight_list=np.array([0])
        for i in range(len(self.weights)):
            tmp_weight_list=np.append(tmp_weight_list,self.weights[i].X)
        tmp_weight_list=tmp_weight_list[1:]
        
        self.weight_record.append(tmp_weight_list)


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
        
        if self.counter==1:
            self.test_roc_list=np.array(testroc)
            self.test_accuracy_list=np.array(accuracy_percentage) 
        else:
            self.test_roc_list=np.append(self.test_roc_list,testroc)
            self.test_accuracy_list=np.append(self.test_accuracy_list,accuracy_percentage)            
        
        
        self.counter=self.counter+1
        
    
    def run_model_with_column_generation(self,plot=False):
        max_iter=len(self.df)-1
        #max_iter=150
        
        
        if plot==True:
            max_iter=15
            for i in range(max_iter):
                self.find_new_column()
                if self.counter==2 and self.ignore_initial==True:
                    for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
                        self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],0)
        
            
                    
                self.solve_problem_with_new_column()
                self.predict_test_data()
            
            
        
        else:
            for i in range(max_iter):
                self.find_new_column()
                if self.counter==2 and self.ignore_initial==True:
                    for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
                        self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],0)
        
            
                    
                self.solve_problem_with_new_column()
                self.predict_test_data()
                
                if self.stopping_condition=="dot_product_rate":
                    cur_obj=self.objective_values[len(self.objective_values)-1]
                    
                    
                    prev=max(self.dot_product_list[len(self.dot_product_list)-2])
                    cur=max(self.dot_product_list[len(self.dot_product_list)-1])

                    if (abs(prev-cur)/prev) < self.stopping_percentage:
                        if i > 1:    
                            break
                    if cur_obj==0:
                        break
                
                
                if self.stopping_condition=="obj":
                    prev_obj=self.objective_values[len(self.objective_values)-2]
                    cur_obj=self.objective_values[len(self.objective_values)-1]
                    
                    if ((prev_obj-cur_obj)/prev_obj) < self.stopping_percentage:
                        break
                    if cur_obj==0:
                        break
                """this condition seems useless.
                if self.stopping_condition=="obj_ratio":
                    cur_obj=self.objective_values[len(self.objective_values)-1]
                    
                    if (cur_obj/self.objective_values[0]) < self.stopping_percentage:
                        break
                    if cur_obj==0:
                        break
                """
                if self.stopping_condition=="accuracy":
                    prev_accu=self.train_accuracy_list[len(self.train_accuracy_list)-2]
                    cur_accu=self.train_accuracy_list[len(self.train_accuracy_list)-1]
                    
                    if ((prev_accu-cur_accu)/prev_accu) < self.stopping_percentage:
                        break
                    if cur_accu==1:
                        break
                #performance was good but it is too complicated to explain.
                if self.stopping_condition=="median_filter":
                    tmp_error=scipy.signal.medfilt(self.errors_list[len(self.errors_list)-1],kernel_size=3)
                    
                    if sum(sum(tmp_error==0)) == tmp_error.shape[0] * tmp_error.shape[1]* 1 :
                    #if sum(sum(tmp_error==0)) > tmp_error.shape[0] * tmp_error.shape[1]* 0.99 :
                        break
    
                if self.stopping_condition=="pos_errors":
                    cur_errors_pos=self.errors_list.sum(axis=2)
                    cur_errors_pos=(cur_errors_pos[len(cur_errors_pos)-1])
                    
                    if sum(cur_errors_pos<=1)/float(len(cur_errors_pos)) > 0.9 :
                        break
                    
                #it is quite interesting.
                if self.stopping_condition=="rank":
                    cur_errors=self.errors_list[len(self.errors_list)-1]
                    tmp_rank=sum(sum(cur_errors<1))/float(cur_errors.shape[0]*cur_errors.shape[1])
    
    
                    prev_errors=self.errors_list[len(self.errors_list)-2]
                    tmp_rank_prev=sum(sum(prev_errors<1))/float(prev_errors.shape[0]*prev_errors.shape[1])
    
                    if len(self.errors_list) == 2:
                            
                        self.rank_log=tmp_rank_prev
                        self.rank_log=np.append(self.rank_log,tmp_rank)
                        
                    if len(self.errors_list) > 2:
                            
                        self.rank_log=np.append(self.rank_log,tmp_rank)
    
    
                    prev_errors=self.errors_list[len(self.errors_list)-2]
                    tmp_rank_prev=sum(sum(prev_errors<1))/float(prev_errors.shape[0]*prev_errors.shape[1])
    
                    
                    if (tmp_rank-tmp_rank_prev)/tmp_rank < self.stopping_percentage:
                    #if (tmp_rank-tmp_rank_prev)/tmp_rank < 0.005:    
                    #if self.objective_values[len(self.objective_values)-1]<=0.1:
                        if self.counter>3: #after ignoring initial train-perf may degrade. To avoid this, I have put additional if.
                            
                            break
                    if tmp_rank==1:
                        break
                
                if self.stopping_condition=="kendall_rank_correlation":
                    no_pairs=len(self.pos)*len(self.neg)
                    count_for_positive=np.sum(self.errors_list>1,axis=2)
                    
                    prev_coeff=(no_pairs-sum(count_for_positive[len(self.errors_list)-2]))/float(no_pairs)
                    curr_coeff=(no_pairs-sum(count_for_positive[len(self.errors_list)-1]))/float(no_pairs)                
                    
                    if (curr_coeff-prev_coeff)/prev_coeff < 0.01:
                        break
                    if curr_coeff==0:
                        break
                                           
    
                if self.stopping_condition=="alpha_trim_symetric":
                    
                    cur_errors_pos=self.errors_list.sum(axis=2)
                    cur_errors_pos=(cur_errors_pos[len(cur_errors_pos)-1])
                    
                    prev_errors_pos=self.errors_list.sum(axis=2)
                    prev_errors_pos=(prev_errors_pos[len(prev_errors_pos)-2])
                    
                    cur_mean=stats.trim_mean(cur_errors_pos, 0.1)
                    prev_mean=stats.trim_mean(prev_errors_pos, 0.1)
                    
                    
                    
                    
                    
                    if cur_mean < 0.05 :
                    #if (prev_mean-cur_mean)/prev_mean < 0.2:
                        break
                
                if self.stopping_condition=="alpha_trim_upper":
                    
                    cur_errors_pos=self.errors_list.sum(axis=2)
                    cur_errors_pos=(cur_errors_pos[len(cur_errors_pos)-1])
                    cur_errors_pos=(cur_errors_pos-cur_errors_pos.min())/float(cur_errors_pos.max()-cur_errors_pos.min())
                    
                    
                    prev_errors_pos=self.errors_list.sum(axis=2)
                    prev_errors_pos=(prev_errors_pos[len(prev_errors_pos)-2])
                    prev_errors_pos=(prev_errors_pos-prev_errors_pos.min())/float(prev_errors_pos.max()-prev_errors_pos.min()) 
                   
                    cur_errors_pos=np.sort(cur_errors_pos)
                    upper_l=int(len(cur_errors_pos)*(1-0.05))
                    cur_errors_pos=cur_errors_pos[0:upper_l]
                    cur_mean=cur_errors_pos.mean()
                    
                    
                    prev_errors_pos=np.sort(prev_errors_pos)
                    upper_l=int(len(prev_errors_pos)*(1-0.05))
                    prev_errors_pos=prev_errors_pos[0:upper_l]
                    prev_mean=prev_errors_pos.mean()
                    
                    
                    #if cur_mean < 0.05 :
                    if (prev_mean-cur_mean)/prev_mean < 0.1:
                        break
                    if cur_mean==0:
                        break
                    
                if self.stopping_condition=="box_plot":
                    
                    cur_errors_pos=self.errors_list.sum(axis=2)
                    cur_errors_pos=(cur_errors_pos[len(cur_errors_pos)-1])
                    cur_errors_pos=(cur_errors_pos-cur_errors_pos.min())/float(cur_errors_pos.max()-cur_errors_pos.min())
                    
                    prev_errors_pos=self.errors_list.sum(axis=2)
                    prev_errors_pos=(prev_errors_pos[len(prev_errors_pos)-2])
                    prev_errors_pos=(prev_errors_pos-prev_errors_pos.min())/float(prev_errors_pos.max()-prev_errors_pos.min()) 
                    
                    IQR=iqr(cur_errors_pos)
                    lower=np.quantile(cur_errors_pos,0.25)
                    upper=np.quantile(cur_errors_pos,0.75)
                    low_thres=lower-1.5*IQR
                    upper_thres=upper+1.5*IQR
                    outliers=cur_errors_pos<=upper_thres
                    cur_mean=cur_errors_pos[outliers].mean()
                    
                    IQR=iqr(prev_errors_pos)
                    lower=np.quantile(prev_errors_pos,0.25)
                    upper=np.quantile(prev_errors_pos,0.75)
                    low_thres=lower-1.5*IQR
                    upper_thres=upper+1.5*IQR
                    outliers=prev_errors_pos<=upper_thres
                    prev_mean=prev_errors_pos[outliers].mean()                
                    
                    if cur_mean < 0.01 :
                        break
                    if (prev_mean-cur_mean)/prev_mean < 0.01:
                        break
                    if cur_mean==0:
                        break
                
                #useless at this concept because error terms's scale alters after each iteration.
                if self.stopping_condition=="chi_sq_dist":
                    
                    cur_errors_pos=self.errors_list.sum(axis=2)
                    cur_errors_pos=(cur_errors_pos[len(cur_errors_pos)-1])
                    
                    prev_errors_pos=self.errors_list.sum(axis=2)
                    prev_errors_pos=(prev_errors_pos[len(prev_errors_pos)-2])
                    
                    if cur_errors_pos.sum() != 0:
    
                        num_bins=int(round(math.sqrt(len(self.pos))))
                        
                        
                        #cur_errors=cur_errors/float(cur_errors.sum())
                        #prev_errors=prev_errors/float(prev_errors.sum())
                        cur_errors_pos=(cur_errors_pos-cur_errors_pos.min())/float(cur_errors_pos.max()-cur_errors_pos.min())
                        prev_errors_pos=(prev_errors_pos-prev_errors_pos.min())/float(prev_errors_pos.max()-prev_errors_pos.min())
                        
                        #bin_l=np.arange(0,1,1/float(num_bins))
                        #bin_l=np.append(bin_l,1)
                        
                        cur_n, bins, patches = plt.hist(cur_errors_pos, num_bins, facecolor='blue', alpha=0.5)
                        prev_n, bins, patches = plt.hist(prev_errors_pos, num_bins, facecolor='blue', alpha=0.5)
                        
                        numerator=(cur_n-prev_n)**2
                        denominator=cur_n+prev_n
                        denominator[denominator==0]=0.1
                        
                        if len(self.errors_list) == 2:
                            
                            self.old_chisq_dist_pos=sum(numerator/denominator)
                            self.new_chisq_dist_pos=sum(numerator/denominator)
                            self.pos_distance_log=self.old_chisq_dist_pos
                        
                        if len(self.errors_list) > 2:
                            
                            self.old_chisq_dist_pos=self.new_chisq_dist_pos
                            self.new_chisq_dist_pos=sum(numerator/denominator)
                            self.pos_distance_log=np.append(self.pos_distance_log,self.new_chisq_dist_pos)
                            
                            diff=abs(self.old_chisq_dist_pos-self.new_chisq_dist_pos)
                            
                            #if diff/self.old_chisq_dist_pos <0.5:
                            if self.new_chisq_dist_pos < 2:
                                break
                    else:
                        break
                        
                    
                

                
                
        
    def get_cumulative_error_for_datapoints(self):
        self.pos_cumul_error=self.errors_list.sum(axis=2)
        self.neg_cumul_error=self.errors_list.sum(axis=1)
    
    def get_conflicted_datapoints(self):
        self.count_for_positive=np.sum(self.errors_list>1,axis=2)
        self.count_for_negative=np.sum(self.errors_list>1,axis=1)
    
    
    
    def solve_full_model(self,time_analysis=False):
        
        if time_analysis==True:
            tmp_points=self.tmp_dist_city.shape[1]
            tmp_array=np.arange(0,tmp_points,1)
            np.random.shuffle(tmp_array)
            tmp_array=tmp_array[:16]
            self.tmp_dist_city=self.tmp_dist_city[:,tmp_array]
        		
        #selected_col_index=0
        self.fm = Model("fsmall_v3")
        
        #no_points=self.used_cols.shape[1]
        no_points=self.tmp_dist_city.shape[1]
        weight_names=[]
        
        for i in range(no_points):
            weight_names.append("w"+str(i))
            #weight_names.append("w"+str(int(self.col_names[1:][i][1:]) ))
            
        
         
        self.fweights=self.fm.addVars(no_points,lb=-1,ub=1,name=weight_names)
        #self.fweights=self.fm.addVars(no_points,lb=-GRB.INFINITY,name=weight_names)
        errors  = self.fm.addVars(len(self.pos),len(self.neg),lb=0.0,name="e")
        
        self.fm.setObjective(sum(errors[f] for f in errors), GRB.MINIMIZE)
        
    
        print ("model constraints are being counstructed")
        #print(datetime.datetime.now())
        
        self.f_constrain=self.fm.addConstrs((errors[i,j] + 
        quicksum(self.fweights[k]*(self.tmp_dist_city[i*len(self.neg)+j,k]) 
        for k in range(len(self.fweights))) >= 1 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c')
        
        
        
        
        print ("end")
        #print(datetime.datetime.now())                    
        
        
        # Solve
        #0 : primal simplex 1:dual simplex  2:barrier
        self.fm.Params.Method=1
        self.fm.Params.Crossover = 0
        self.fm.optimize()
        #m.getVars()
        
        i=0
        for i in range(len(self.fweights)):
            tmp=weight_names[i]
            
            if i==0:
                self.fcol_list=np.array(int(tmp[1:]))
                self.fweight_list=np.array(self.fweights[i].X)
            else:
                self.fcol_list=np.append(self.fcol_list,int(tmp[1:]))
                self.fweight_list=np.append(self.fweight_list,self.fweights[i].X)
    
        
        train_class_numpy=self.train_class.values
        res_train=np.dot(self.data_distance_numpy[:,self.fcol_list],self.fweight_list)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
        #accuracy
        self.fclf=tree.DecisionTreeClassifier(max_depth=1)
        self.fclf.fit(res_with_class.memb.values.reshape(len(res_with_class),1),res_with_class.trainclass.values.reshape(len(res_with_class),1))
        train_predict=self.fclf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        self.ftrain_accuracy=accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        #accuracyends    
        
        self.ftrainroc=roc_auc_score(res_with_class.trainclass,res_with_class.memb)
        


    def predict_test_full_model(self):
        
        test_class_numpy=np.array(self.test_class)
        test_data_numpy=self.test_data.values
           
        te_tr_distance=pd.DataFrame(distance_matrix(test_data_numpy, self.train_data), index=self.test_data.index)
        te_tr_distance=te_tr_distance.values
        te_tr_distance = (te_tr_distance - self.mean_to_scale_test) / (self.sd_to_scale_test)    
        res=np.dot(te_tr_distance[:,self.fcol_list],self.fweight_list)
        res=res.reshape(len(res),1)
        
        res_with_class=pd.DataFrame({'testclass':test_class_numpy[:,0],'memb':res[:,0]},index=range(len(res)))
        #clf=DecisionTreeClassifier(max_depth=1)
        
        
        self.ftestroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
        
        #print res_with_class
        
        test_predict=self.fclf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        self.ftest_accu=accuracy_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        
    
    def solve_integer_first_time(self):
        
        
        tmp_binary=np.zeros(self.tmp_dist_city.shape[0])
        cur_errors=self.errors_list[len(self.errors_list)-1].reshape(len(self.pos)*len(self.neg),1)
        cur_errors=cur_errors[:,0]
        
        tmp_binary[(cur_errors<1-self.epsilon_int)]=int(1)



        self.m_int = Model("integer_small")
        
        #used_cols keep the columns used in the model.
        #remained_cols keep the list of columns that can be included to the model.
        
        #first column is selected randomly for now.
        #selected_col_index=0
        
        w_name="iw"+self.col_names[self.selected_col_index]
       
        self.int_weights=self.m_int.addVars(1,lb=-1,ub=1,name=w_name)
        self.int_weights.start=self.weights[0].X
        
        self.x_i  = self.m_int.addVars(len(self.pos),len(self.neg),lb=0.0,ub=1.0,vtype=GRB.BINARY,name="x")
        #self.x_i.start=tmp_binary.reshape(len(self.pos),len(self.neg))
        self.m_int.setObjective(sum(self.x_i[x] for x in self.x_i), GRB.MAXIMIZE)
        
        #for k in range(len(weights)):
         #   print k
        
        #pair_counter=0
        """for p in errors:
          m.addConstr(quicksum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
          pair_counter=pair_counter+1
        """
        #m.addConstrs(sum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
         
        
        print ("model constraints are being counstructed")
        print(datetime.datetime.now())
        
        
        M=100
        
        self.int_constrain=self.m_int.addConstrs((M*(1-self.x_i[i,j]) - self.epsilon_int + 
        quicksum(self.int_weights[k]*(self.tmp_dist_city[i*len(self.neg)+j,self.selected_col_index]) 
        for k in range(len(self.weights))) >= 0 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c_int')
        
        self.obj_lower_bound_cnstr=self.m_int.addConstr(sum(self.x_i[x] for x in self.x_i)>=sum(tmp_binary))
        
        
        print ("end")
        print(datetime.datetime.now())                    
        self.m_int.update()
        self.m_int.write('/Users/can/Desktop/columngeneration/int_deneme.lp')
        
        # Solve
        
        
        #0 : primal simplex 1:dual simplex
        #self.m_int.Params.Method=0
        self.m_int.optimize()
        self.m_int.getVars()
        """
        self.errors_list=np.zeros(1*len(self.pos)*len(self.neg))
        self.errors_list=self.errors_list.reshape(1,len(self.pos),len(self.neg))
        
        for i in range (len(self.pos)):
            for j in range (len(self.neg)):
               self.errors_list[0,i,j]=errors[i,j].X
        """
        train_class_numpy=self.train_class.values
        res_train=np.dot(self.data_distance_numpy[:,self.selected_col_index],self.int_weights[0].X)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
        #accuracy
        self.clf=tree.DecisionTreeClassifier(max_depth=1)
        self.clf.fit(res_with_class.memb.values.reshape(len(res_with_class),1),res_with_class.trainclass.values.reshape(len(res_with_class),1))
        train_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        train_accuracy=accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        #accuracyends    
        
        trainroc=roc_auc_score(res_with_class.trainclass,res_with_class.memb)
        
        #self.duals=np.array(self.m.Pi)
        
        self.int_train_roc_list=np.array(trainroc)
        self.int_train_accuracy_list=np.array(train_accuracy)
        self.int_objective_values=np.array(self.m.objVal)
        
        self.int_weight_record=list(np.array([[0]]))
        
        self.int_weight_record.append(np.array([self.int_weights[0].X]))
        
        
    def int_predict_test(self):
        
        
        np_weights=np.array(self.int_weights[0].X)
        if (self.counter>2):
            for i in range(self.counter-2):
                np_weights=np.append(np_weights,self.int_weights[i+1].X)  
        
        
        res=np.dot(self.distance_btw_test_and_selected,np_weights)
        res=res.reshape(len(res),1)
        
        test_class_numpy=np.array(self.test_class)
        res_with_class=pd.DataFrame({'testclass':test_class_numpy[:,0],'memb':res[:,0]},index=range(len(res)))
        #clf=DecisionTreeClassifier(max_depth=1)
        
        
        testroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
        
        #print res_with_class
        
        test_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        accuracy_percentage=accuracy_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        
        if self.counter==2:
            self.int_test_roc_list=np.array(testroc)
            self.int_test_accuracy_list=np.array(accuracy_percentage) 
        else:
            self.int_test_roc_list=np.append(self.int_test_roc_list,testroc)
            self.int_test_accuracy_list=np.append(self.int_test_accuracy_list,accuracy_percentage)            
        
        
        
class rankSVM_column_generation_grid:
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,distance="euclidian",stopping_condition="obj",stopping_percentage=0.01,selected_col_index=0,scale=True,ignore_initial=False,manual_selection=False,manuel_select_counter=0,epsilon_int=0.001,p=0.5,grid=True,convergence_percentage=0.0001,use_median_to_initialize=False,learning_rate=0.001):
        self.train_data=train_data
        self.train_class=train_class
        self.test_data=test_data
        self.test_class=test_class
        self.df=df
        self.df_test=df_test
        self.distance=distance
        self.stopping_condition=stopping_condition
        self.stopping_percentage=stopping_percentage
        self.selected_col_index=selected_col_index
        self.scale=scale
        self.counter=1
        self.ignore_initial=ignore_initial
        self.manual_selection=manual_selection
        self.manuel_select_counter=manuel_select_counter
        self.epsilon_int=epsilon_int
        self.p=p
        self.grid=grid
        self.convergence_percentage=convergence_percentage
        self.use_median_to_initialize=use_median_to_initialize
        self.lr=learning_rate
    def data_preprocess(self):
        
        np.random.seed(0)
        data_matrix=self.train_data.values
        self.dual_list=[]
        self.data_limits=np.array([np.min(data_matrix,axis=0)])
        self.data_limits=self.data_limits.transpose()
        self.data_limits=np.append(self.data_limits,np.array([np.max(data_matrix,axis=0)]).transpose(),axis=1)
        self.data_limits = self.data_limits.astype('float32') 
        limits=self.data_limits
        
        tmp_class_data=self.df[['class']]
        tmp_class_data['counter'] = range(len(self.df))
        tmp_df=self.df.drop(['class'], axis=1)
    
        self.pos=tmp_class_data.loc[tmp_class_data['class']==1]
        self.neg=tmp_class_data.loc[tmp_class_data['class']==-1]
        self.pos=self.pos.iloc[:,1].values
        self.neg=self.neg.iloc[:,1].values
    
        import itertools
        pairs = list(itertools.product(self.pos, self.neg))
        data_matrix=tmp_df.values
        
        
        if self.distance=="euclidian":
            self.data_distance=calc_pnorm_dist(np.array([data_matrix[self.selected_col_index,:]]),data_matrix,self.p,"euclidian")
            self.full_data_distance=calc_pnorm_dist(data_matrix,data_matrix,self.p,"euclidian")
        elif self.distance=="pnorm":
            self.data_distance=calc_pnorm_dist(np.array([data_matrix[self.selected_col_index,:]]),data_matrix,self.p,"pnorm")
        else:
            print("not available")
        self.pairs_distance_dif_table=pd.DataFrame(pairs,columns=['pos_sample','neg_sample'])
        
        self.pairs_check=self.pairs_distance_dif_table
        
        dimension=(len(self.pairs_distance_dif_table),1)
        self.pairs_distance_dif_table=self.pairs_distance_dif_table.values
        self.tmp_dist_city=np.zeros(dimension)
        self.data_distance_numpy=self.data_distance
        tmp_dim=(len(self.pairs_distance_dif_table),len(data_matrix))
        self.full_tmp_dist=np.zeros(tmp_dim)
        
        #NO SCALING!!!!
        self.mean_to_scale_test=np.mean(self.data_distance_numpy,axis=0)
        self.sd_to_scale_test=np.std(self.data_distance_numpy,axis=0)    
        if self.scale==True:    
            self.data_distance_numpy = (self.data_distance_numpy - self.mean_to_scale_test) / (self.sd_to_scale_test)
            #self.full_data_distance = (self.full_data_distance - np.mean(self.full_data_distance,axis=0)) / (np.std(self.full_data_distance,axis=0) )
       
        for i in range(len(pairs)):
            #print cntr
            index_pos=self.pairs_distance_dif_table[i,0]
            index_neg=self.pairs_distance_dif_table[i,1]
            tmp_dif=self.data_distance_numpy[index_pos,:] - self.data_distance_numpy[index_neg,:]
            self.tmp_dist_city[i,:]=tmp_dif
            full_tmp_dif=self.full_data_distance[index_pos,:] - self.full_data_distance[index_neg,:]
            self.full_tmp_dist[i,:]=full_tmp_dif            
        
        #mean_to_scale_test=np.mean(tmp_dist_city,axis=0)
        #sd_to_scale_test=np.std(tmp_dist_city,axis=0)    
            
        #tmp_dist_city = (tmp_dist_city - tmp_dist_city.mean()) / (tmp_dist_city.std())
        
        self.training_data_index=tmp_df.index.values
        self.col_names=[]
        
        for i in range(self.tmp_dist_city.shape[1]):
            self.col_names.append( "p" + str(i))
        
        
        self.number_of_pairs=len(self.tmp_dist_city)
        self.tmp_dist_city_correlation=sum(self.tmp_dist_city>0)/float(self.tmp_dist_city.shape[0])
        
        
        pos_data=self.train_data.values[self.pos,:]
        neg_data=self.train_data.values[self.neg,:]
        self.pos_neg_pairs=np.array([ x for x in itertools.product(pos_data,neg_data) ])
        self.location_convergence_obj=[]
        self.record_obj_values=[]
        
    def solve_problem_first_time(self):
        #selected_col_index=0
        self.m = Model("small_v4")
        self.m.setParam('OutputFlag',False)
        
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
        
        self.used_cols_name=np.array(self.col_names[self.selected_col_index])
        self.used_cols_name=np.append(self.used_cols_name,self.col_names[self.selected_col_index])
        
        self.weights=self.m.addVars(1,lb=-1,ub=1,name=w_name)
        #self.weights=self.m.addVars(1,lb=-GRB.INFINITY,name=w_name)
        errors  = self.m.addVars(len(self.pos),len(self.neg),lb=0.0,name="e")
        
        self.m.setObjective(sum(errors[f] for f in errors), GRB.MINIMIZE)
        
        #for k in range(len(weights)):
         #   print k
        
        #pair_counter=0
        """for p in errors:
          m.addConstr(quicksum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
          pair_counter=pair_counter+1
        """
        #m.addConstrs(sum(tmp_dist_city.iloc[pair_counter,w]*weights[w] for w in weights) + errors[p] >= 1)
         
        
        print ("model constraints are being counstructed")
        print(datetime.datetime.now())
        
        self.constrain=self.m.addConstrs((errors[i,j] + 
        quicksum(self.weights[k]*(self.tmp_dist_city[i*len(self.neg)+j,self.selected_col_index]) 
        for k in range(len(self.weights))) >= 1 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c')
        
        
        
        
        print ("end")
        print(datetime.datetime.now())                    
        
        
        # Solve
        
        
        #0 : primal simplex 1:dual simplex
        self.m.Params.Method=0
        start_time=datetime.datetime.now()
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_difference=(end_time-start_time).seconds
        self.m.getVars()
        
        self.errors_list=np.zeros(1*len(self.pos)*len(self.neg))
        self.errors_list=self.errors_list.reshape(1,len(self.pos),len(self.neg))
        
        for i in range (len(self.pos)):
            for j in range (len(self.neg)):
               self.errors_list[0,i,j]=errors[i,j].X
        
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
        
        self.duals=np.array(self.m.Pi)
        self.dual_list.append(self.duals)
        
        self.train_roc_list=np.array(trainroc)
        self.train_accuracy_list=np.array(train_accuracy)
        self.objective_values=np.array(self.m.objVal)
        
        self.weight_record=list(np.array([[0]]))
        
        self.weight_record.append(np.array([self.weights[0].X]))


    def predict_test_data(self):
        test_class_numpy=np.array(self.test_class)
        test_data_numpy=self.test_data.values
        if self.counter==1:
            self.focused_point_list=[]
            focused_point=np.array([self.train_data.values[self.selected_col_index,:]])
            self.focused_point_list.append(focused_point[0,:])
            self.distance_btw_test_and_selected=calc_pnorm_dist(focused_point,self.test_data.values,self.p,self.distance)
            if self.scale==True:    
                self.distance_btw_test_and_selected = (self.distance_btw_test_and_selected - self.mean_to_scale_test) / (self.sd_to_scale_test)
        else:
            if self.grid==False:
                focused_point=np.array([self.train_data.values[self.selected_col_index,:]])
            else: 
                focused_point=np.array([self.new_point])
            self.focused_point_list.append(focused_point[0,:])
            tmp_dist=calc_pnorm_dist(focused_point,self.test_data.values,self.p,self.distance)
            if self.scale==True:    
                tmp_dist = (tmp_dist - self.mean_to_scale_test) / (self.sd_to_scale_test)
            self.distance_btw_test_and_selected=np.append(self.distance_btw_test_and_selected,tmp_dist,axis=1)

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
        
        if self.counter==1:
            self.test_roc_list=np.array(testroc)
            self.test_accuracy_list=np.array(accuracy_percentage) 
        else:
            self.test_roc_list=np.append(self.test_roc_list,testroc)
            self.test_accuracy_list=np.append(self.test_accuracy_list,accuracy_percentage)            
        
        
        self.counter=self.counter+1
        
    def find_new_point_from_grid(self):
        
        self.location_convergence_obj=[]
        
        #this uses existing points.
        res_dot_product=np.array(abs((self.full_tmp_dist.T).dot(self.duals)))
        if(self.counter==2):
            self.max_res_dot=max(res_dot_product)
        self.current_res_dot=max(res_dot_product)
        
        record_objective=np.array(max(res_dot_product))
        #index_of_p=np.argmax(res_dot_product)
        #print(max(res_dot_product))
        #raw_input("press enter to continue")
        
        #this uses median.
        tmp=calc_pnorm_dist((self.focused_point_list[len(self.focused_point_list)-1]).reshape(1,self.train_data.shape[1]),self.train_data.values,self.p,self.distance)
        index_of_p=np.where(tmp==np.sort(tmp,axis=0)[len(tmp)//2])[0][0]
        location_of_init_point=self.train_data.values[index_of_p,:]
        
        limits=self.data_limits
        A=self.pos_neg_pairs
        dual_vars=self.duals
        
        #this part was trying to find a point. However, Kuban hoca said that it will be always on corners.
        #when lambdas change the selected corner may change but it will always locate on the corners.
        
        


        tf.reset_default_graph()
        
        no_of_points=A.shape[0]
        batch_size=no_of_points
        
        pos_samp = tf.placeholder(tf.float32, shape=[batch_size, 1,limits.shape[0]], name='x_pos_samp')
        neg_samp = tf.placeholder(tf.float32, shape=[batch_size, 1,limits.shape[0]], name='x_neg_samp')
        dual_cons= tf.placeholder(tf.float32, shape=[batch_size],name='x_dual_cons')
        dual_cons=tf.reshape(dual_cons,[batch_size,1])
        #this part was trying to find a point. However, Kuban hoca said that it will be always on corners.
        #when lambdas change the selected corner may change but it will always locate on the corners.
        
        
        weights = tf.Variable(tf.zeros([1,limits.shape[0]]), dtype=tf.float32, name="weights")
        
        tmp_tf=-1*tf.math.abs(tf.reduce_sum(tf.math.multiply(dual_cons,tf.norm(pos_samp-weights,ord='euclidean',axis=2))-tf.math.multiply(dual_cons,tf.norm(neg_samp-weights,ord='euclidean',axis=2))))
        
        tvars = tf.trainable_variables()
        weight_vars = [var for var in tvars if 'weights' in var.name]
        
        
        #grad_desc_obj = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        #grad_desc_obj = tf.train.AdamOptimizer(learning_rate=0.01)
        grad_desc_obj = tf.train.AdamOptimizer(learning_rate=self.lr)
        trainer_grad=grad_desc_obj.minimize(tmp_tf,var_list=weight_vars)
        
        
        """
        #optimize with constraints
        # Clipping operation. 
        max_W_0 = weights[0].assign(tf.maximum(limits[0,0], weights[0]))
        min_W_0 = weights[0].assign(tf.minimum(limits[0,1], weights[0]))
        
        max_W_1 = weights[1].assign(tf.maximum(limits[1,0], weights[1]))
        min_W_1 = weights[1].assign(tf.minimum(limits[1,1], weights[1]))
        
        clip = tf.group(max_W_0,min_W_0,max_W_1,min_W_1)
        #clip end
        """
        
        start_time=datetime.datetime.now()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if self.use_median_to_initialize==True:
            for i in range(limits.shape[0]):
                assign_op=weights[0,i].assign(location_of_init_point[i]+np.random.normal(0,0.1,1)[0])
                sess.run(assign_op)
        #print(sess.run(weights))
        #raw_input("press enter to continue")
        no_epoch = 1000
        how_many_with_this_batch_size=int(no_of_points/batch_size)
        counter=0
        obj_value=np.array(0)
        
        for i in range(no_epoch):
            #print i
            counter=0
            nums=np.array([tmp for tmp in range(no_of_points)])
            random.shuffle(nums)
            obj_per_sample=np.zeros(no_of_points)
            
            for j in range(how_many_with_this_batch_size):
                #print j
                focused=nums[counter:(counter+batch_size)]
                real_point=A[focused,]
                real_point_pos=real_point[:,0,]
                real_point_pos  = real_point_pos.reshape(batch_size,1,limits.shape[0])
                real_point_neg=real_point[:,1,]
                real_point_neg  = real_point_neg.reshape(batch_size,1,limits.shape[0])
                real_dual=dual_vars[focused]
                real_dual=real_dual.reshape(batch_size,1)
                
                _,obj_follow =sess.run([trainer_grad,tmp_tf],feed_dict={pos_samp:real_point_pos,neg_samp:real_point_neg,dual_cons:real_dual})
          
                obj_per_sample[counter]= obj_follow              
                counter=counter+batch_size
            #print(np.sum(obj_per_sample))
            obj_value=np.append(obj_value,np.sum(obj_per_sample))
            record_objective=np.append(record_objective,np.sum(obj_per_sample))
            
            if i>2:
                if(abs((obj_value[len(obj_value)-1]-obj_value[len(obj_value)-2])/obj_value[len(obj_value)-2])<self.convergence_percentage):
                    #print("stopped due to convergence percentage")
                    break
        end_time=datetime.datetime.now()
        self.opt_difference=self.opt_difference+(end_time-start_time).seconds
        self.location_convergence_obj.append(record_objective)   
        #print(self.location_convergence_obj)
        #raw_input("press enter to continue")
        #self.location_convergence_obj.append(obj_value[1:])   
        learnt_W = sess.run(weights)       
        
        #print(obj_value[len(obj_value)-1])
        #print(obj_value)
        
        self.convergence_list=obj_value
        self.new_point=learnt_W.reshape(1*limits.shape[0])

        
    def solve_problem_with_new_column(self):

        self.w_name="w"+str(self.counter)
        
        
        
        self.weights[len(self.weights)] = self.m.addVar(lb=-1, ub=1,name=self.w_name)
        #self.weights[len(self.weights)] = self.m.addVar(lb=-GRB.INFINITY,name=self.w_name)
        
        focused_point=np.array([self.new_point])
        tmp=calc_pnorm_dist(focused_point,self.train_data.values,self.p,self.distance)
        
        self.mean_to_scale_test=np.mean(tmp,axis=0)
        self.sd_to_scale_test=np.std(tmp,axis=0)    
        if self.scale==True:    
            tmp = (tmp - self.mean_to_scale_test) / (self.sd_to_scale_test)
        
        
        self.data_distance_numpy=np.append(self.data_distance_numpy,tmp,axis=1)
        used_cols_tmp=np.zeros((self.used_cols.shape[0],1))
        for i in range(self.used_cols.shape[0]):
            #print cntr
            index_pos=self.pairs_distance_dif_table[i,0]
            index_neg=self.pairs_distance_dif_table[i,1]
            tmp_dif=tmp[index_pos,:] - tmp[index_neg,:]
            used_cols_tmp[i,:]=tmp_dif
        
        
        self.used_cols=np.append(self.used_cols,used_cols_tmp, axis=1)
        self.used_cols_name=np.append(self.used_cols_name,self.w_name)
        
        for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
            self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],self.used_cols[i*len(self.neg)+j,self.used_cols.shape[1]-1])
        
        
        
        
        self.m.update()
        #self.m.write('/Users/can/Desktop/columngeneration/svm_second.lp')
        self.m.Params.Method=1
        start_time=datetime.datetime.now()
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_difference=self.opt_difference+(end_time-start_time).seconds
        self.m.getVars()
        self.duals=np.array(self.m.Pi)
        self.dual_list.append(self.duals)
                
        
        for i in range(len(self.weights)):
            tmp=self.used_cols_name[i+1]
            
            if i==0:
                self.col_list=np.array(int(tmp[1:]))
                self.weight_list=np.array(self.weights[i].X)
            else:
                self.col_list=np.append(self.col_list,int(tmp[1:]))
                self.weight_list=np.append(self.weight_list,self.weights[i].X)
                
                
        train_class_numpy=self.train_class.values
        #res_train=np.dot(method1.data_distance_numpy[:,self.col_list],self.weight_list)
        res_train=np.dot(self.data_distance_numpy,self.weight_list)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
         #accuracy
        self.clf=tree.DecisionTreeClassifier(max_depth=1)
        self.clf.fit(res_with_class.memb.values.reshape(len(res_with_class),1),res_with_class.trainclass.values.reshape(len(res_with_class),1))
        train_predict=self.clf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        train_accuracy=accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        #accuracyends
        
        
        trainroc=roc_auc_score(res_with_class.trainclass,res_with_class.memb)
        
        
        self.train_roc_list=np.append(self.train_roc_list,trainroc)
        self.train_accuracy_list=np.append(self.train_accuracy_list,train_accuracy)
        self.objective_values=np.append(self.objective_values,self.m.objVal)
        
        tmp_weight_list=np.array([0])
        for i in range(len(self.weights)):
            tmp_weight_list=np.append(tmp_weight_list,self.weights[i].X)
        tmp_weight_list=tmp_weight_list[1:]
        
        self.weight_record.append(tmp_weight_list)     
        

    def run_model_with_column_generation(self,plot=False,rep=0,fold=0):
        max_iter=3*len(self.df)
        #max_iter=150
        
        if plot==True:
            max_iter=15
            for i in range(max_iter):
                self.find_new_point_from_grid()
                if self.counter==2 and self.ignore_initial==True:
                    for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
                        self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],0)
        
            
                    
                self.solve_problem_with_new_column()
                self.predict_test_data()
                if(self.train_roc_list[len(self.train_roc_list)-1]==1):
                    i=150
                
            
            
        else:
            for i in range(max_iter):
                if i==0:
                    self.prev_res_dot=0.00001
                else:
                    self.prev_res_dot=self.current_res_dot
                self.find_new_point_from_grid()
                
                #use this part to record survival local convergence
                obj_list=[]
                obj_list.append(self.location_convergence_obj)
                #os.chdir("/Users/can/Desktop/results/survival_detailed")
                obj_conv=pd.DataFrame(obj_list)
                tmp_name="rep"+str(rep)+"fold_"+str(fold)+"iter_no_"+str(i)+".csv"
                #obj_conv.to_csv(tmp_name,index=False)
                ####
                if self.counter==2 and self.ignore_initial==True:
                    for i,j in itertools.product(range(len(self.pos)), range(len(self.neg))):
                        self.m.chgCoeff(self.constrain[i,j],self.weights[len(self.weights)-1],0)
        
            
                    
                self.solve_problem_with_new_column()
                self.predict_test_data()
                if i>2: 
                    if self.stopping_condition=="obj":
                        prev_obj=self.objective_values[len(self.objective_values)-2]
                        cur_obj=self.objective_values[len(self.objective_values)-1]
                        
                        if ((prev_obj-cur_obj)/prev_obj) < self.stopping_percentage:
                            break
                        if cur_obj==0:
                            break
                        
                    if self.stopping_condition=="dot_product":
                        cur_obj=self.objective_values[len(self.objective_values)-1]
                        
                        if ((self.current_res_dot)/self.max_res_dot) < self.stopping_percentage:
                            break
                        if cur_obj==0:
                            break
                    if self.stopping_condition=="dot_product_rate":
                        cur_obj=self.objective_values[len(self.objective_values)-1]
                        
                        if (abs(self.current_res_dot-self.prev_res_dot)/self.prev_res_dot) < self.stopping_percentage:
                            break
                        if cur_obj==0:
                            break

                    
class rankSVM_prototype:
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,distance="euclidian",stopping_condition="obj",stopping_percentage=0.01,selected_col_index=0,scale=True,ignore_initial=False,manual_selection=False,manuel_select_counter=0,epsilon_int=0.001,p=0.5,grid=True,convergence_percentage=0.0001,how_many_prototype=10,learning_rate=0.00001,epoch=False):
        self.train_data=train_data
        self.train_class=train_class
        self.test_data=test_data
        self.test_class=test_class
        self.df=df
        self.df_test=df_test
        self.distance=distance
        self.stopping_condition=stopping_condition
        self.stopping_percentage=stopping_percentage
        self.selected_col_index=selected_col_index
        self.scale=scale
        self.counter=1
        self.ignore_initial=ignore_initial
        self.manual_selection=manual_selection
        self.manuel_select_counter=manuel_select_counter
        self.epsilon_int=epsilon_int
        self.p=p
        self.grid=grid
        self.convergence_percentage=convergence_percentage
        self.how_many_prototype=how_many_prototype
        self.lr=learning_rate
        self.stop_only_epoch=epoch
    def data_preprocess(self):
        
        data_matrix=self.train_data.values
        self.data_limits=np.array([np.min(data_matrix,axis=0)])
        self.data_limits=self.data_limits.transpose()
        self.data_limits=np.append(self.data_limits,np.array([np.max(data_matrix,axis=0)]).transpose(),axis=1)
        self.data_limits = self.data_limits.astype('float32') 
        limits=self.data_limits
        
        tmp_class_data=self.df[['class']]
        tmp_class_data['counter'] = range(len(self.df))
        tmp_df=self.df.drop(['class'], axis=1)
    
        self.pos=tmp_class_data.loc[tmp_class_data['class']==1]
        self.neg=tmp_class_data.loc[tmp_class_data['class']==-1]
        self.pos=self.pos.iloc[:,1].values
        self.neg=self.neg.iloc[:,1].values
    
        import itertools
        pairs = list(itertools.product(self.pos, self.neg))
        data_matrix=tmp_df.values
        
        
        if self.distance=="euclidian":
            self.data_distance=calc_pnorm_dist(np.array([data_matrix[self.selected_col_index,:]]),data_matrix,self.p,"euclidian")
        elif self.distance=="pnorm":
            self.data_distance=calc_pnorm_dist(np.array([data_matrix[self.selected_col_index,:]]),data_matrix,self.p,"pnorm")
        else:
            print("not available")
        self.pairs_distance_dif_table=pd.DataFrame(pairs,columns=['pos_sample','neg_sample'])
        
        self.pairs_check=self.pairs_distance_dif_table
        
        dimension=(len(self.pairs_distance_dif_table),1)
        self.pairs_distance_dif_table=self.pairs_distance_dif_table.values
        self.tmp_dist_city=np.zeros(dimension)
        self.data_distance_numpy=self.data_distance
        
        """
        #NO SCALING!!!!
        self.mean_to_scale_test=np.mean(self.data_distance_numpy,axis=0)
        self.sd_to_scale_test=np.std(self.data_distance_numpy,axis=0)    
        if self.scale==True:    
            self.data_distance_numpy = (self.data_distance_numpy - self.mean_to_scale_test) / (self.sd_to_scale_test)
        """
       
        for i in range(len(pairs)):
            #print cntr
            index_pos=self.pairs_distance_dif_table[i,0]
            index_neg=self.pairs_distance_dif_table[i,1]
            tmp_dif=self.data_distance_numpy[index_pos,:] - self.data_distance_numpy[index_neg,:]
            self.tmp_dist_city[i,:]=tmp_dif
            
        
        #mean_to_scale_test=np.mean(tmp_dist_city,axis=0)
        #sd_to_scale_test=np.std(tmp_dist_city,axis=0)    
            
        #tmp_dist_city = (tmp_dist_city - tmp_dist_city.mean()) / (tmp_dist_city.std())
        
        self.training_data_index=tmp_df.index.values
        self.col_names=[]
        
        for i in range(self.tmp_dist_city.shape[1]):
            self.col_names.append( "p" + str(i))
        
        
        self.number_of_pairs=len(self.tmp_dist_city)
        self.tmp_dist_city_correlation=sum(self.tmp_dist_city>0)/float(self.tmp_dist_city.shape[0])
        
        
        pos_data=self.train_data.values[self.pos,:]
        neg_data=self.train_data.values[self.neg,:]
        self.pos_neg_pairs=np.array([ x for x in itertools.product(pos_data,neg_data) ])
        
        
    def prototype_optimize(self):
        limits=self.data_limits
        A=self.pos_neg_pairs
        #dual_vars=method1.duals
        #how_many_prototype=30
        
        tf.reset_default_graph()
        
        no_of_points=A.shape[0]
        batch_size=no_of_points
        
        
        pos_samp = tf.placeholder(tf.float32, shape=[batch_size, 1,limits.shape[0]], name='x_pos_samp')
        neg_samp = tf.placeholder(tf.float32, shape=[batch_size, 1,limits.shape[0]], name='x_neg_samp')
        
        
        prototypes= tf.Variable(tf.truncated_normal([self.how_many_prototype,limits.shape[0]], stddev=0.05), dtype=tf.float32 ,name='prototypes')
        weights = tf.Variable(tf.truncated_normal([self.how_many_prototype,1], stddev=0.05), dtype=tf.float32, name="weights")
        
        """
        tmp_tf=tf.reduce_sum(tf.math.multiply(tf.math.minimum(
                (tf.matmul(tf.norm(pos_samp-prototypes,ord='euclidean',axis=2),weights)
                -tf.matmul(tf.norm(neg_samp-prototypes,ord='euclidean',axis=2),weights))
                ,0),-1))
        """        
        
        tmp_tf=tf.reduce_sum(tf.math.multiply(tf.nn.relu(
                -1*(tf.matmul(tf.norm(pos_samp-prototypes,ord='euclidean',axis=2),weights)
                -tf.matmul(tf.norm(neg_samp-prototypes,ord='euclidean',axis=2),weights))
                ),1))
        
        tvars = tf.trainable_variables()
        weight_vars = [var for var in tvars if 'weights' in var.name]
        prot_vars = [var for var in tvars if 'prototypes' in var.name]
        
        #grad_desc_obj = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        grad_desc_obj = tf.train.AdamOptimizer(learning_rate=self.lr)
        trainer_grad=grad_desc_obj.minimize(tmp_tf,var_list=tvars)
        
        
        """
        #optimize with constraints
        # Clipping operation. 
        max_W_0 = weights[0].assign(tf.maximum(limits[0,0], weights[0]))
        min_W_0 = weights[0].assign(tf.minimum(limits[0,1], weights[0]))
        
        max_W_1 = weights[1].assign(tf.maximum(limits[1,0], weights[1]))
        min_W_1 = weights[1].assign(tf.minimum(limits[1,1], weights[1]))
        
        clip = tf.group(max_W_0,min_W_0,max_W_1,min_W_1)
        #clip end
        """
        
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        no_epoch = 50000
        how_many_with_this_batch_size=int(no_of_points/batch_size)
        counter=0
        self.obj_value=np.array(0)
        
        self.tr_roc_list=np.array(0)
        self.te_roc_list=np.array(0)
        
        self.weight_list=list(np.array([0]))
        
        for i in range(no_epoch):
            #print (i)
            counter=0
            nums=np.array([tmp for tmp in range(no_of_points)])
            random.shuffle(nums)
            obj_per_sample=np.zeros(no_of_points)
            
            for j in range(how_many_with_this_batch_size):
                #print j
                focused=nums[counter:(counter+batch_size)]
                real_point=A[focused,]
                real_point_pos=real_point[:,0,]
                real_point_pos  = real_point_pos.reshape(batch_size,1,limits.shape[0])
                real_point_neg=real_point[:,1,]
                real_point_neg  = real_point_neg.reshape(batch_size,1,limits.shape[0])
                #real_dual=dual_vars[focused]
                #real_dual=real_dual.reshape(batch_size,1)
                
                _,obj_follow =sess.run([trainer_grad,tmp_tf],feed_dict={pos_samp:real_point_pos,neg_samp:real_point_neg})
                #print(obj_follow)
                obj_per_sample[counter]= obj_follow              
                counter=counter+batch_size
            self.obj_value=np.append(self.obj_value,np.sum(obj_per_sample))
            
            learnt = sess.run(tvars)
            learnt_prot=learnt[0]
            learnt_weights=learnt[1]
            self.weight_list.append(learnt_weights)
            #if i%100==0:
                #print(learnt_weights)
            
            
            tr_dist=calc_pnorm_dist(learnt_prot,self.train_data.values,self.p,self.distance)
            tr_predict=np.matmul(tr_dist,learnt_weights)
            res_with_class=pd.DataFrame({'testclass':(self.train_class.values)[:,0],'memb':tr_predict[:,0]},index=range(len(tr_predict)))
            trainroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
            self.tr_roc_list=np.append(self.tr_roc_list,trainroc)
            
            
            test_dist=calc_pnorm_dist(learnt_prot,self.test_data.values,self.p,self.distance)
            test_predict=np.matmul(test_dist,learnt_weights)
            res_with_class=pd.DataFrame({'testclass':(self.test_class.values)[:,0],'memb':test_predict[:,0]},index=range(len(test_predict)))
            testroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
            self.te_roc_list=np.append(self.te_roc_list,testroc)
            if self.stop_only_epoch==False:
                if i>2:    
                    if abs((self.obj_value[len(self.obj_value)-2]-self.obj_value[len(self.obj_value)-1])/self.obj_value[len(self.obj_value)-2])<self.convergence_percentage:
                        #print(self.obj_value[len(self.obj_value)-1])
                        break

            
        learnt = sess.run(tvars)
        self.learnt_prot=learnt[0]
        self.learnt_weights=learnt[1]
        
        
        tr_dist=calc_pnorm_dist(self.learnt_prot,self.train_data.values,self.p,self.distance)
        tr_predict=np.matmul(tr_dist,self.learnt_weights)
        res_with_class=pd.DataFrame({'testclass':(self.train_class.values)[:,0],'memb':tr_predict[:,0]},index=range(len(tr_predict)))
        self.trainroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
        
        
        test_dist=calc_pnorm_dist(learnt_prot,self.test_data.values,self.p,self.distance)
        test_predict=np.matmul(test_dist,self.learnt_weights)
        res_with_class=pd.DataFrame({'testclass':(self.test_class.values)[:,0],'memb':test_predict[:,0]},index=range(len(test_predict)))
        self.testroc=roc_auc_score(res_with_class.testclass,res_with_class.memb)
        #learnt_W.reshape(1*limits.shape[0])
        

        

        
