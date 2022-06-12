"""
    srcg_dec_lr_exp_smooth: constant learning rate with exponential smoothing
"""
import numpy as np
from cg.scripts.algs.base_srcg import base_srcg


class srcg_dec_lr_exp_smooth(base_srcg):
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,
                      distance="euclidian",stopping_condition=None,
                      stopping_percentage=None,lr=None,
                      selected_col_index=0,scale=True,alpha=0.1):
        
     
        
        super(srcg_dec_lr_exp_smooth,self).__init__(train_data,train_class,test_data,test_class,df,df_test,
                          distance=distance,stopping_condition=stopping_condition,
                          stopping_percentage=stopping_percentage,lr=lr,
                          selected_col_index=0,scale=scale)
        
        self.alpha=alpha
    
    
    def reference_weights(self):
        
        if self.counter == 2:
            self.weight_memory=self.weight_record[-1]
        elif self.counter > 2:
            self.weight_memory=(self.weight_memory)*(self.alpha) +\
                                self.weight_record[-1][:-1]*(1-self.alpha)
            
            self.weight_memory=np.concatenate([self.weight_memory,np.array([self.weight_record[-1][-1]])])

        return self.weight_memory
    
    def schedule_lr(self):
        self.lr=self.lr_init*(self.counter**0.5)