#smooth_ranking_cg
import pandas as pd
import datetime
import math
import numpy as np
from gurobipy import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree,svm
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

from cg.scripts.algs.ranking_cg import ranking_cg
import tensorflow as tf

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
            elif dist_type =="pnorm":
                raise NotImplementedError()

    return result



class ranking_cg_prototype_unb(ranking_cg):
    
    def __init__(self,train_data,train_class,
                      test_data,test_class,
                      df,df_test,distance="euclidian",
                      stopping_condition=None,stopping_percentage=0.01,
                      lr=0,
                      selected_col_index=0,scale=True,prot_stop_perc=1e-5,max_epoch=1000):
        
        
        super(ranking_cg_prototype_unb,self).__init__(train_data,train_class,test_data,test_class,df,df_test,
                          distance=distance,stopping_condition=stopping_condition,
                          stopping_percentage=stopping_percentage,lr=lr,
                          selected_col_index=0,scale=scale)

        self.data_preprocess()
        
        data_matrix=self.train_data.values
        self.data_limits=np.array([np.min(data_matrix,axis=0)])
        self.data_limits=self.data_limits.transpose()
        self.data_limits=np.append(self.data_limits,np.array([np.max(data_matrix,axis=0)]).transpose(),axis=1)
        self.data_limits = self.data_limits.astype('float32') 
        
        self.pos_data=self.train_data.values[self.pos,:]
        self.neg_data=self.train_data.values[self.neg,:]
        self.pos_neg_pairs=np.array([ x for x in itertools.product(self.pos_data,self.neg_data) ])

        self.pos_data=tf.constant(self.pos_neg_pairs[:,0,:].astype("float32"))
        self.neg_data=tf.constant(self.pos_neg_pairs[:,1,:].astype("float32"))
        
        self.prot_stop_perc=prot_stop_perc
        self.model_lr=lr
        self.max_epoch=max_epoch
        #self.model_optimizer = tf.keras.optimizers.Adam(
        #    learning_rate=self.model_lr)

        #self.pos_data=tf.constant(self.pos_data.astype("float32"))
        #self.neg_data=tf.constant(self.neg_data.astype("float32"))
        self.rng = np.random.default_rng(12345)
        self.data_distance_numpy=np.expand_dims(self.data_distance_numpy[:,selected_col_index],axis=1)
        
    # #@tf.function
    # def get_loss(self,duals,weights):
        
    #     first=tf.norm(self.pos_data-weights,axis=1)
    #     sec=tf.norm(self.neg_data-weights,axis=1)
        
    #     #return tf.abs(tf.matmul(np.expand_dims(duals,axis=0),tf.expand_dims(first-sec,axis=1)))
    #     return -1*tf.abs(tf.reduce_sum(tf.multiply(duals,first-sec)))
    
    
    def find_new_column(self):
        
        
        @tf.function
        def get_loss(duals,weights):
            
            first=tf.norm(self.pos_data-weights,axis=1)
            sec=tf.norm(self.neg_data-weights,axis=1)
            
            #return tf.abs(tf.matmul(np.expand_dims(duals,axis=0),tf.expand_dims(first-sec,axis=1)))
            return -1*tf.abs(tf.reduce_sum(tf.multiply(duals,first-sec)))
        
        iter_counter=0
        #a=np.array([[2,3,-2],[-1,0,-3]])
        
        start_time=datetime.datetime.now()
        self.location_convergence_obj=[1e-6]
        res_dot_product=np.array(abs((self.full_tmp_dist.T).dot(self.duals)))
    
        if(self.counter==2):
            self.max_res_dot=max(res_dot_product)
        self.current_res_dot=max(res_dot_product)   
    
        record_objective=np.array(max(res_dot_product))
        
        index_of_p=np.argmax(res_dot_product)
        location_of_init_point=self.train_data.values[index_of_p,:]
        
        #this uses median initialization.
        #tmp=calc_pnorm_dist((self.focused_point_list[len(self.focused_point_list)-1]).reshape(1,self.train_data.shape[1]),self.train_data.values,self.p,self.distance)
        #index_of_p=np.where(tmp==np.sort(tmp,axis=0)[len(tmp)//2])[0][0]
        #location_of_init_point=self.train_data.values[index_of_p,:]
        
        limits=self.data_limits
        A=self.pos_neg_pairs
        dual_vars=self.duals.astype("float32")
        
        
        

        #tf.reset_default_graph()
        
        
        no_of_points=A.shape[0]
        batch_size=no_of_points
        
        #pos=tf.constant(self.pos_data)
        #neg=tf.constant(self.neg_data)
        
        weights = tf.Variable(location_of_init_point+self.rng.normal(0,0.1,len(location_of_init_point)), dtype=tf.float32, trainable=True, name="weights")
        stopper=True
        best=location_of_init_point
        best_obj=0
        while stopper:
            with tf.GradientTape() as tape:
                
                #loss=self.get_loss(dual_vars,weights)
                loss=get_loss(dual_vars,weights)
                #loss=self.graph_loss(dual_vars,weights)
                #diff=self.graph_loss(weights)
                #loss=-1*tf.abs(tf.reduce_sum(tf.multiply(dual_vars,diff)))
                self.location_convergence_obj.append(loss.numpy())
            grads=tape.gradient(loss,weights)
            
            self.model_optimizer.apply_gradients(zip([grads],[weights]))
            prev=self.location_convergence_obj[-2]
            cur=self.location_convergence_obj[-1]
            if abs(cur-prev)/abs(prev) < self.prot_stop_perc or iter_counter==self.max_epoch:
                #print(self.location_convergence_obj)
                stopper=False
            iter_counter+=1
            if cur < best_obj:
                best_obj=cur
                best=weights.numpy()
        end_time=datetime.datetime.now()
        #self.new_point=weights.numpy()
        #if abs(self.location_convergence_obj[1])<abs(cur):
        #    self.new_point=weights.numpy()
        #    print("better point is found")
        #else:
        #    self.new_point=location_of_init_point
        self.new_point=best
        self.opt_time+=(end_time-start_time).seconds
        print("The best objective is %f:\nThe initial objective is %f:"%(best_obj,self.location_convergence_obj[1]))
        
    def solve_problem_with_new_column(self):

        
        weight_vals=[self.weights[i].X for i in range(len(self.weights))]
        
        names_to_retrieve=[]
        for i in range(len(self.pos)):
            for j in range(len(self.neg)):
                names_to_retrieve.append("e["+str(i)+","+str(j)+"]")
                
        error_vars=[]
        
        error_vars.append([self.m.getVarByName(name) for name in names_to_retrieve])


        self.w_name="w"+str(self.counter)
        
        
        
        #self.weights[len(self.weights)] = self.m.addVar(lb=-1, ub=1,name=self.w_name)
        self.weights[len(self.weights)] = self.m.addVar(lb=-GRB.INFINITY,name=self.w_name)
        
        focused_point=np.array([self.new_point])
        tmp=calc_pnorm_dist(focused_point,self.train_data.values,-1,self.distance)
        
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
        
        
        for t in range (len(self.weights)-1):
           self.weights[t].PStart=weight_vals[t]
        self.weights[len(self.weights)-1].PStart=0  
        
        iter_num=self.errors_list.shape[0]-1
        for i in range (len(self.pos)):
            for j in range (len(self.neg)):
                error_vars[0][i*len(self.neg)+j].PStart=self.errors_list[iter_num,i,j]
        
        #self.m.write('/Users/can/Desktop/columngeneration/svm_second.lp')
        self.m.Params.Method=0#It was 1 before
        self.m.Params.LPWarmStart=2#1
        start_time=datetime.datetime.now()
        self.m.optimize()
        end_time=datetime.datetime.now()
        self.opt_time+=(end_time-start_time).seconds
        
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
        #res_train=np.dot(self.data_distance_numpy[:,self.col_list],self.weight_list)
        res_train=np.dot(self.data_distance_numpy,self.weight_list)
        res_train=res_train.reshape(len(res_train),1)
        
        res_with_class=pd.DataFrame({'trainclass':train_class_numpy[:,0],'memb':res_train[:,0]},index=range(len(res_train)))
         #accuracy
        self.clf=tree.DecisionTreeClassifier(max_depth=1)
        #self.clf=svm.LinearSVC(class_weight="balanced")
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
    
        
    def predict_test_data(self):
        test_class_numpy=np.array(self.test_class)
        test_data_numpy=self.test_data.values
        if self.counter==1:
            self.focused_point_list=[]
            focused_point=np.array([self.train_data.values[self.selected_col_index,:]])
            self.focused_point_list.append(focused_point[0,:])
            self.distance_btw_test_and_selected=calc_pnorm_dist(focused_point,self.test_data.values,-1,self.distance)
            if self.scale==True:    
                self.distance_btw_test_and_selected = (self.distance_btw_test_and_selected - self.mean_to_scale_test[0]) / (self.sd_to_scale_test[0])
        else:
    
            focused_point=np.array([self.new_point])
            self.focused_point_list.append(focused_point[0,:])
            tmp_dist=calc_pnorm_dist(focused_point,self.test_data.values,-1,self.distance)
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
        self.test_predictions=test_predict

    
    def run(self,plot=False,name=None):
        
        # #self.lr=self.lr_init
        # def _loss(duals,weights):
            
        #     #duals=tf.constant(duals)
        #     first=tf.norm(self.pos_data-weights,axis=1)
        #     sec=tf.norm(self.neg_data-weights,axis=1)
        #     #print(duals)
        #     #return tf.abs(tf.matmul(np.expand_dims(duals,axis=0),tf.expand_dims(first-sec,axis=1)))
        #     return -1*tf.abs(tf.reduce_sum(tf.multiply(duals,first-sec)))
        #     #return first-sec
        
        # #self.data_preprocess() #taken care of inside the constructor.
        # self.graph_loss=tf.function(_loss)
        self.solve_problem_first_time()
        self.predict_test_data()
        
        stopper=True
        i=2
        while stopper:
            tf.keras.backend.clear_session()
            self.model_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_lr)
            self.find_new_column()
            #self.schedule_lr()
            self.solve_problem_with_new_column()
            self.predict_test_data()
            if plot==True:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                #ax.scatter(x=self.df_test.f0, y=self.df_test.f1, c=self.test_predictions,alpha=0.01)
                pos_pred=self.test_predictions==1
                neg_pred=self.test_predictions==-1
                ax.scatter(x=self.df_test.f0[pos_pred], y=self.df_test.f1[pos_pred], color='green',alpha=0.01)
                ax.scatter(x=self.df_test.f0[neg_pred], y=self.df_test.f1[neg_pred], color='purple',alpha=0.02)
                
                pos_idx=self.df['class']==1
                neg_idx=self.df['class']==-1
                #ax.scatter(x=self.df.f0, y=self.df.f1, c=self.df['class'])
                ax.scatter(x=self.df.f0[pos_idx], y=self.df.f1[pos_idx], color='green',marker='o',label='+ class')
                ax.scatter(x=self.df.f0[neg_idx], y=self.df.f1[neg_idx], color='purple',marker='v',label='- class')
                #prototypes
                prot_loc=np.concatenate([np.expand_dims(m,axis=0) for m in self.focused_point_list])
                ax.scatter(x=prot_loc[:,0],y=prot_loc[:,1],c='red',marker='x',label='prototype')
                ax.set_xlabel('feature 1')
                ax.set_ylabel('feature 2')
                #ax.set_ylim((0,15))
                ax.legend(loc="lower right")
                #fig
                address="/Users/can/Desktop/ranking_cg_extension/plots/"
                fig.savefig(address+name+str(i)+".png")
                i+=1
            stopper=self.stopping_criteria()
        

            
        
        