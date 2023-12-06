from cg.scripts.algs.srcg import srcg
from cg.scripts.algs.srcg_prototype import srcg_prototype
from cg.scripts.algs.srcg_decayinglr import srcg_decayinglr
from cg.scripts.algs.srcg_exp_smooth import srcg_exp_smooth
from cg.scripts.algs.srcg_dec_lr_exp_smooth import srcg_dec_lr_exp_smooth
from cg.scripts.algs.l1_rank import l1_rank
from cg.scripts.algs.full_rank import full_rank
from cg.scripts.algs.l1_rank_cg import l1_rank_cg
from cg.scripts.algs.l_inf_rank import l_inf_rank
from cg.scripts.algs.ranking_cg import ranking_cg
from cg.scripts.algs.ranking_cg_prototype import ranking_cg_prototype
from cg.scripts.algs.ranking_cg_prototype_unb import ranking_cg_prototype_unb

""""Interface to algorithms"""

def init_alg(alg_type,train_data,train_class,test_data,test_class,df,df_test,
                          distance,stopping_condition,
                          stopping_percentage,lr, alpha,
                          selected_col_index,scale,prot_stop_perc=None,max_epoch=None,prototype_lr=0.001):
    #selected_col_index: the first column that we added manually to initiate CG iterations.
    
    if alg_type == "base":
        # no smoothing no change in learning rate.
        return srcg(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    elif alg_type == "srcg_prototype":
        # no smoothing but learning rate is scaled by iteration number.
        return srcg_prototype(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale,prot_stop_perc=prot_stop_perc,
                                  max_epoch=max_epoch,prototype_lr=prototype_lr)
    
    
    elif alg_type == "dec_lr":
        # no smoothing but learning rate is scaled by iteration number.
        return srcg_decayinglr(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    elif alg_type == "exp_smooth":
        # no smoothing but learning rate is scaled by iteration number.
        return srcg_exp_smooth(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale,
                                  alpha=alpha)
    elif alg_type == "dec_lr_exp_smooth":
        # no smoothing but learning rate is scaled by iteration number.
        return srcg_dec_lr_exp_smooth(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale,
                                  alpha=alpha)
    elif alg_type == "l1_rank":
        # no smoothing but learning rate is scaled by iteration number.
        return l1_rank(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    elif alg_type == "l1_rank_cg":
        # no smoothing but learning rate is scaled by iteration number.
        return l1_rank_cg(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    elif alg_type == "l_inf_rank":
        # no smoothing but learning rate is scaled by iteration number.
        return l_inf_rank(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    elif alg_type == "ranking_cg":
        # no smoothing but learning rate is scaled by iteration number.
        return ranking_cg(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=selected_col_index,scale=scale)
    elif alg_type == "ranking_cg_prototype":
        # no smoothing but learning rate is scaled by iteration number.
        return ranking_cg_prototype(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=prototype_lr,
                                  selected_col_index=selected_col_index,scale=scale,prot_stop_perc=prot_stop_perc,
                                  max_epoch=max_epoch)
    
    elif alg_type == "ranking_cg_prototype_unb":
        # no smoothing but learning rate is scaled by iteration number.
        return ranking_cg_prototype_unb(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=prototype_lr,
                                  selected_col_index=selected_col_index,scale=scale,prot_stop_perc=prot_stop_perc,
                                  max_epoch=max_epoch)
    
    
    elif alg_type == "full_rank":
        # no smoothing but learning rate is scaled by iteration number.
        return full_rank(train_data,train_class,test_data,test_class,df,df_test,
                                  distance=distance,stopping_condition=stopping_condition,
                                  stopping_percentage=stopping_percentage,lr=lr,
                                  selected_col_index=0,scale=scale)
    else:
        raise NotImplementedError
    