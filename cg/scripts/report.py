#srcg main.py
import matplotlib.pyplot as plt
import sys
import numpy as np
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


import random
#%%

df,df_test,test_class,test_data,train_class,train_data=selected_data_set(datasetname=d_name,location=data_location)
random.seed(3)
data=train_data.append(test_data)           
class_data=train_class.append(test_class)
#%%


from cg.scripts.algs.init_alg import init_alg

lr_list=[0,0.1,0.5,1.0,2.0,5.0,10.0,100.0]


alg_type="base"
stp_perc=0.1
stp_cond="tr_obj"
#lr=1.0
alpha=0.1

base_test=[]

for lr in lr_list:
    method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                              distance="euclidian",stopping_condition=stp_cond,
                              stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                              selected_col_index=0,scale=True)

    method1.run()
    base_test.append(method1.test_roc_list)
#%%

alg_type="l1_rank"
l1_rank_test=[]

for lr in lr_list:
    method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                              distance="euclidian",stopping_condition=stp_cond,
                              stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                              selected_col_index=0,scale=True)

    method1.run()
    #l1_rank_test.append(method1.test_roc_list)
    l1_rank_test.append(np.repeat(method1.test_roc_list[0],20))
    

#%%

alg_type="l1_rank_cg"
l1_rank_cg_test=[]

for lr in lr_list:
    method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                              distance="euclidian",stopping_condition=stp_cond,
                              stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                              selected_col_index=0,scale=True)

    method1.run()
    l1_rank_cg_test.append(method1.test_roc_list)

#%%

alg_type="dec_lr"
dec_lr_test=[]

for lr in lr_list:
    method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                              distance="euclidian",stopping_condition=stp_cond,
                              stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                              selected_col_index=0,scale=True)

    method1.run()
    dec_lr_test.append(method1.test_roc_list)
#%%
alg_type="exp_smooth"
exp_smooth_test=[]

for lr in lr_list:
    method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                              distance="euclidian",stopping_condition=stp_cond,
                              stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                              selected_col_index=0,scale=True)

    method1.run()
    exp_smooth_test.append(method1.test_roc_list)

#%%

def test_auc_plot(measure_list,lr_list,p_name,y_name="Test AUC"):
    
    fig, ax = plt.subplots()
    
    for measure,lr in zip(measure_list,lr_list):
        
        ax.plot(np.arange(len(measure)),measure,label="lambda="+str(lr))
    ax.set_title(y_name+" vs lambda (Algorithm "+p_name+")")
    ax.legend(loc="lower right")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(y_name)
    ax.set_xlim((-1,20))
    ax.set_ylim((0.80,1.0))
    
#%%
test_auc_plot(base_test,lr_list,p_name="base")  


test_auc_plot(l1_rank_cg_test,lr_list,p_name="l1_rank_cg")  

test_auc_plot(dec_lr_test,lr_list,p_name="dec_lr")  


test_auc_plot(exp_smooth_test,lr_list,p_name="exp_smooth")  

test_auc_plot(l1_rank_test,lr_list,p_name="l1_rank")  



    
    

#print(method1.test_roc_list)
#print(method1.objective_values)
#print(method1.real_training_objective)
