"""
    srcg: constant learning rate with no exponential smoothing
"""
from cg.scripts.algs.base_srcg import base_srcg
import numpy as np
import pandas as pd
from gurobipy import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy.spatial import distance_matrix
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class full_rank(base_srcg):
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,
                      distance="euclidian",stopping_condition=None,
                      stopping_percentage=None,lr=None,
                      selected_col_index=0,scale=True):
        
     
        
        super(full_rank,self).__init__(train_data,train_class,test_data,test_class,df,df_test,
                          distance=distance,stopping_condition=stopping_condition,
                          stopping_percentage=stopping_percentage,lr=lr,
                          selected_col_index=0,scale=scale)
    
    
    #def reference_weights(self):
    #    return np.zeros(len(self.weight_record[-1])) 
    
    def schedule_lr(self):
        self.lr=self.lr_init
        
    
    def run(self,plot=False,name=None):
        """
        in l1_rank model no column generation takes place. We optimize full model
        with l1 penalty.
        """
        #pre-process data 
        self.data_preprocess()
        
        #set regularization parameter
        self.schedule_lr()
        
        #construct model
        self.fm = Model("full_model")
        
        
        no_points=self.tmp_dist_city.shape[1]
        weight_names=[]
        
        for i in range(no_points):
            weight_names.append("w"+str(i))
            
        
        
        self.fweights=self.fm.addVars(no_points,lb=-1,ub=1,name=weight_names)
        #self.absweights=self.fm.addVars(no_points,lb=0.0,name="abs")
        #self.fweights=self.fm.addVars(no_points,lb=-GRB.INFINITY,name=weight_names)
        errors  = self.fm.addVars(len(self.pos),len(self.neg),lb=0.0,name="e")
        
        
        obj=(quicksum(errors[f] for f in errors))
        
        self.fm.setObjective(obj, GRB.MINIMIZE)
        
        #constraints
        
        
        self.f_constrain=self.fm.addConstrs((errors[i,j] + 
        quicksum(self.fweights[k]*(self.tmp_dist_city[i*len(self.neg)+j,k]) 
        for k in range(len(self.fweights))) >= 1 for i, j in 
        itertools.product(range(len(self.pos)), 
        range(len(self.neg)))),name='c')
        
        
        #self.fm.addConstrs((self.absweights[i]==abs_(self.fweights[i]) for i in range(no_points)),name="tmp")
        #self.abs_pos=self.fm.addConstrs((self.absweights[i] - self.fweights[i] >= 0.0 for i in range (no_points)),name="aaa")
        #self.abs_neg=self.fm.addConstrs((self.absweights[i] + self.fweights[i] >= 0.0 for i in range (no_points)),name="bbb")
        
        
        
        self.fm.Params.Method=1
        #self.fm.Params.Crossover = 0
        self.fm.optimize()
        
        
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
        self.train_accuracy_list=[accuracy_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)]
                
        self.train_roc_list=[roc_auc_score(res_with_class.trainclass,res_with_class.memb)]        
        
        #print(method1.real_training_objective)
        
        self.objective_values=[self.fm.objVal]
        
        trainsense=sensitivity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainspec=specificity_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        traingeo=geometric_mean_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainprec=precision_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        trainfone=f1_score(res_with_class.trainclass.values.reshape(len(res_with_class),1), train_predict)
        
        self.train_sensitivity_list=[trainsense]
        self.train_specificity_list=[trainspec]
        self.train_geometric_mean_list=[traingeo]
        self.train_precision_list=[trainprec]
        self.train_fone_list=[trainfone]
        
        err_counter=0
        record=np.zeros(len(self.pos)*len(self.neg))
        for x in self.fm.getVars():
            if x.VarName.find("e")!=-1:
                record[err_counter]=x.X
                err_counter+=1
        record=record.reshape(1,len(self.pos),len(self.neg))
        
        
        self.real_training_objective=[np.sum(record)]
        
        
        
        ## training related calculations are done. Calculate test performance.
        
        test_class_numpy=np.array(self.test_class)
        test_data_numpy=self.test_data.values
           
        te_tr_distance=pd.DataFrame(distance_matrix(test_data_numpy, self.train_data), index=self.test_data.index)
        if self.distance == "sq_euclidian":
            te_tr_distance=te_tr_distance**2
        
        te_tr_distance=te_tr_distance.values
        if self.scale==True: 
            te_tr_distance = (te_tr_distance - self.mean_to_scale_test) / (self.sd_to_scale_test)    
        res=np.dot(te_tr_distance[:,self.fcol_list],self.fweight_list)
        res=res.reshape(len(res),1)
        
        res_with_class=pd.DataFrame({'testclass':test_class_numpy[:,0],'memb':res[:,0]},index=range(len(res)))
        #clf=DecisionTreeClassifier(max_depth=1)
        
        
        self.test_roc_list=[roc_auc_score(res_with_class.testclass,res_with_class.memb)]
        
        #print res_with_class
        
        test_predict=self.fclf.predict(res_with_class.memb.values.reshape(len(res_with_class),1))
        self.test_accuracy_list=[accuracy_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)]
        
        
        tesense=sensitivity_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tespec=specificity_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tegeo=geometric_mean_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        teprec=precision_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        tefone=f1_score(res_with_class.testclass.values.reshape(len(res_with_class),1), test_predict)
        
        self.test_sensitivity_list=[tesense]
        self.test_specificity_list=[tespec]
        self.test_geometric_mean_list=[tegeo]
        self.test_precision_list=[teprec]
        self.test_fone_list=[tefone]
        
        
        self.weight_record=[self.fweight_list]
        self.test_predictions=test_predict
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
            ax.set_xlabel('feature 1')
            ax.set_ylabel('feature 2')
            #ax.set_ylim((0,15))
            ax.legend(loc="lower right")
            #fig
            address="/Users/can/Desktop/ranking_cg_extension/plots/"
            fig.savefig(address+name+".png")
        
        
        