#srcg main.py

import sys
folder_location="/Users/can/Documents/GitHub/Ranking-CG"
data_location=folder_location+"/cg/data"
sys.path.append(folder_location)
#please type the location where github_column_generation folder exists.


#dataset name
d_name="xor" #parabol_3

#folder_location=folder_location+"/github_column_generation"
#data_location=folder_location+"/data"
#script_location=folder_location+"/scripts"


from cg.scripts.read_available_datasets import selected_data_set
from cg.scripts.algs.init_alg import init_alg

import random




"""
7 lines below must be commented out to use GPU.
"""
# import tensorflow as tf
# import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



#208- 479 - 447-602-175-165-85-87
#%%
df,df_test,test_class,test_data,train_class,train_data=selected_data_set(datasetname=d_name,location=data_location)

#df=df.sort_values('class')
#train_class=train_class.loc[df.index]
#train_data=train_data.loc[df.index]


random.seed(3)
data=train_data.append(test_data)           
class_data=train_class.append(test_class)

#%%
#random.seed(10)
#df=df.drop(df.loc[(df['class']==-1),].sample(frac=.87,random_state=10).index)


import matplotlib.pyplot as plt
plt.scatter(x=df.f0,y=df.f1,c=df['class'])

#train_data=df[['f0','f1']]
#train_class=df[['class']]


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
                                ranking_cg_prototype
                                full_rank
                               
                               
                               
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
                        In ranking_cg_prototype, it is the lr of Adam optimizer.
   
               
   
    alpha             : Controls the exponential smoothing of past weights. 
                        Required in exo_smooth and dec_lr_exp_smooth
                        
    prot_stop_perc    : If the change ratio between consecutive iterations of 
                        ranking_cg_prototype is less than this value, iterations are
                        ceased
                        
    max_epoch         : max number of iterations to optimize the non-linear subproblem
                        of ranking cg.
"""





alg_type="srcg_prototype"
stp_perc=0.001#0.01
stp_cond="tr_obj"
lr=1.0
alpha=0.1
prot_stop_perc=1e-5
max_epoch=1000


method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance="euclidian",stopping_condition=stp_cond,
                          stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                          selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                          max_epoch=max_epoch)

method1.run(plot=False,name="acircles_ranking_cg_prototype_")
#method1.run()

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
#method1.opt_time # in seconds.

##prototype points list in ranking_cg_prototype.
#method1.focused_point_list

#other metrics
#print(method1.test_sensitivity_list)
#print(method1.test_specificity_list)
#print(method1.test_geometric_mean_list)
#print(method1.test_precision_list)
#print(method1.test_fone_list)

#%%

#plot the focused points

import matplotlib.pyplot as plt
import numpy as np

prot_loc=np.concatenate([np.expand_dims(m,axis=0) for m in method1.focused_point_list[:11]])
#prot_loc=prot_loc[:-3,:]


fig, ax = plt.subplots()

ax.scatter(x=df.f0, y=df.f1, c=df['class'])
ax.scatter(x=prot_loc[:,0],y=prot_loc[:,1],c='red')
#ax.set_xlim([-0.5,1.5])
#ax.set_ylim([-1,1.4])
fig

#%%

#plot the decision boundaries.

import matplotlib.pyplot as plt
import numpy as np



fig, ax = plt.subplots()

ax.scatter(x=df_test.f0, y=df_test.f1, c=method1.test_predictions,alpha=0.05)
ax.scatter(x=df.f0, y=df.f1, c=df['class'])
#ax.scatter(x=prot_loc[:,0],y=prot_loc[:,1],c='red')
fig

