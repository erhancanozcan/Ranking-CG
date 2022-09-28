#srcg main.py

import sys
folder_location="/Users/can/Documents/GitHub/Ranking-CG"
data_location=folder_location+"/cg/data"
sys.path.append(folder_location)
#please type the location where github_column_generation folder exists.


#dataset name
d_name="xor"

#folder_location=folder_location+"/github_column_generation"
#data_location=folder_location+"/data"
#script_location=folder_location+"/scripts"


from cg.scripts.read_available_datasets import selected_data_set
from cg.scripts.algs.init_alg import init_alg

import random



#208- 479 - 447-602-175-165-85-87
#%%
df,df_test,test_class,test_data,train_class,train_data=selected_data_set(datasetname=d_name,location=data_location)
random.seed(3)
data=train_data.append(test_data)           
class_data=train_class.append(test_class)
#%%

#from cg.scripts.algs.init_alg import init_alg

"""
Parameters:
   
    Alg_type         : Options
                                base
                                dec_lr
                                exp_smooth
                                dec_lr_exp_smooth
                                l1_rank
                                l1_rank_cg
                                l_inf_rank
                                ranking_cg
                               
                               
                               
    stp_perc          : controls the column generation iterations. Required for all algorithms.
                        Please note that this is a critical parameter in raking-cg Experiments.
                        Please make sure that stp_cond is tr_obj whenever ranking_cg is used.
   
    stp_cond         : the rule stopping cg iterations.
                      Options
                              real_tr_obj : considers only the sum xi_{p,n} values. 
                              
                              tr_obj      : considers the surrogate objective we are minimizing. 
                                                   the sum xi_{p,n} + deviation penalizer.
                               
                              tr_roc      : considers calculated tr roc values. We expect this
                                            to be similar to real_tr_obj. Also, I believe that
                                            it is easier to explain the first two options.
                                            
       
   
   
    lr                : controls the magnitude of gradient steps. 
                        In base, it remains constant.
                        In decaying_lr: it increases after each iteration.
                        In l1_rank, it is the penalty coefficient.
                        In l1_rank_cg, it is the penalty coefficient.
                        In l_inf_rank, it is the penalty coefficient.
   
               
   
    alpha             : Controls the exponential smoothing of past weights. 
                        Required in exo_smooth and dec_lr_exp_smooth
"""





alg_type="ranking_cg"
stp_perc=0.01
stp_cond="tr_obj"
lr=1.0
alpha=0.1


method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance="euclidian",stopping_condition=stp_cond,
                          stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                          selected_col_index=0,scale=True)

method1.run()

#All the statistics that we want to check.
#print(method1.test_roc_list)
#print(method1.train_roc_list)

#print(method1.test_accuracy_list)
#print(method1.train_accuracy_list)

#print(method1.objective_values)
#print(method1.real_training_objective)

#learned weights
#method1.weight_record[-1]

#optimization_time
#method1.opt_time

#%%







