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

#%%
df,df_test,test_class,test_data,train_class,train_data=selected_data_set(datasetname=d_name,location=data_location)

#df=df.sort_values('class')
#train_class=train_class.loc[df.index]
#train_data=train_data.loc[df.index]


random.seed(3)
data=train_data.append(test_data)           
class_data=train_class.append(test_class)

#%%


import matplotlib.pyplot as plt
scatter_pos=df['class'] == 1
scatter_neg=df['class'] == -1
fig, ax = plt.subplots(figsize=(6.4,4.8))
ax.scatter(x=df.loc[scatter_pos,:].f0,y=df.loc[scatter_pos,:].f1,label="Positive Instance",marker="v")
ax.scatter(x=df.loc[scatter_neg,:].f0,y=df.loc[scatter_neg,:].f1,label="Negative Instance",marker="X")
ax.legend(loc="upper center")
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_ylim((-0.6,2.0))
ax.set_title('XOR')               
#fig
address="/Users/can/Desktop/ranking_cg_extension/plots/"
name="ScatterPlotTestAUCComparison"
#fig.savefig(address+name+str(i)+".png")
fig.savefig(address+name+".svg")




#%%

alg_type="ranking_cg_prototype_unb"
stp_perc=30#0.01
stp_cond="num_f"
lr=0.01#0.01 if we scale the objective by length.
alpha=0.1
prot_stop_perc=1e-5
max_epoch=1000


method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance="euclidian",stopping_condition=stp_cond,
                          stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                          selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                          max_epoch=max_epoch)

method1.run(plot=False,name="acircles_ranking_cg_prototype_")

ranking_cg_prot_unb_test_AUC=method1.test_roc_list

#%%


alg_type="ranking_cg_prototype"

method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance="euclidian",stopping_condition=stp_cond,
                          stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                          selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                          max_epoch=max_epoch)

method1.run(plot=False,name="acircles_ranking_cg_prototype_")

ranking_cg_prot_test_AUC=method1.test_roc_list

#%%



alg_type="srcg_prototype"

method1=init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance="euclidian",stopping_condition=stp_cond,
                          stopping_percentage=stp_perc,lr=lr, alpha=alpha,
                          selected_col_index=0,scale=True,prot_stop_perc=prot_stop_perc,
                          max_epoch=max_epoch)

method1.run(plot=False,name="acircles_ranking_cg_prototype_")

srcg_prot_test_AUC=method1.test_roc_list

#print(method1.objective_values)
#print(method1.real_training_objective)


#%%



import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(6.4,4.8))

ax.plot(np.arange(len(ranking_cg_prot_test_AUC)),ranking_cg_prot_test_AUC,label="Ranking-CG Prototype")
ax.plot(np.arange(len(ranking_cg_prot_unb_test_AUC)),ranking_cg_prot_unb_test_AUC,label="Unbounded Ranking-CG Prototype")
ax.plot(np.arange(len(srcg_prot_test_AUC)),srcg_prot_test_AUC,label="Smooth Ranking-CG Prototype")
ax.legend(loc="lower right")
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('AUC')
ax.set_title('Test AUC vs Number of Iterations')               
#fig
address="/Users/can/Desktop/ranking_cg_extension/plots/"
name="Test AUC Comparison"
#fig.savefig(address+name+str(i)+".png")
fig.savefig(address+name+".svg")



