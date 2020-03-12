library(data.table)

folder="/Users/can/desktop/github_column_generation/experiment_results"


models<-c("rank_svm","rank_cg","rank_cg_prot_dot_product_rate","rank_full_model")
m<-c("/rank_svm","/ranking_cg/obj_0.005","/ranking_cg_prototype/median_initial/learning_rate_001/obj_0.005","/rank_full_model")
file_name_extension<-c("_rank_svm.csv","_rank_cg.csv","_rank_cg_prototype.csv","_rank_full_svm.csv")
#dataset<-c("xor","xor_two","votes","ionosphere","sonar","spectf","survival","cancer_wbc","spambase","parkinsons","cleveland_heart","xor_curse_of_d1","xor_curse_of_d2","xor_curse_of_d3","monks1","monks1_v2","survival_scaled")
dataset<-c("xor","votes","ionosphere","sonar","spectf","survival_scaled","cancer_wbc","monks1","parkinsons","cleveland_heart")



mean_test_roc_summary<-matrix(0,nrow=length(dataset),ncol=5)

mean_test_roc_summary<-as.data.table(mean_test_roc_summary)
mean_test_roc_summary$V1<-as.factor(mean_test_roc_summary$V1)

for (i in 1:length(dataset)){
  mean_test_roc_summary[i,1]=dataset[i]
  for (j in 1:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    
    data<-read.csv(tmp_file)
      
    mean_test_roc_summary[i,j+1]=round(mean(data$test_roc),3)
    

  }
}
colnames(mean_test_roc_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prot_dot_product_rate","rank_full_model")






sd_summary<-matrix(0,nrow=length(dataset),ncol=5)

sd_summary<-as.data.table(sd_summary)
sd_summary$V1<-as.factor(sd_summary$V1)

for (i in 1:length(dataset)){
  sd_summary[i,1]=dataset[i]
  for (j in 1:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    
    
    data<-read.csv(tmp_file)
      
    sd_summary[i,j+1]=round(sd(data$test_roc),3)
    
  }
}
colnames(sd_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prot_dot_product_rate","rank_full_model")



avg_no_features_summary<-matrix(0,nrow=length(dataset),ncol=5)

avg_no_features_summary<-as.data.table(avg_no_features_summary)
avg_no_features_summary$V1<-as.factor(avg_no_features_summary$V1)

for (i in 1:length(dataset)){
  avg_no_features_summary[i,1]=dataset[i]
  for (j in 2:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    data<-read.csv(tmp_file)
    
    avg_no_features_summary[i,j+1]=round(mean(data$Num_features),3)
  }
}
colnames(avg_no_features_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prototype_dot_product_rate","rank_full_model")
avg_no_features_summary$rank_svm<-avg_no_features_summary$rank_full_model





train_roc_summary<-matrix(0,nrow=length(dataset),ncol=5)

train_roc_summary<-as.data.table(train_roc_summary)
train_roc_summary$V1<-as.factor(train_roc_summary$V1)

for (i in 1:length(dataset)){
  train_roc_summary[i,1]=dataset[i]
  for (j in 1:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    data<-read.csv(tmp_file)
    
    train_roc_summary[i,j+1]=round(mean(data$train_roc),3)
  }
}
colnames(train_roc_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prototype","rank_full_model")




test_accu_summary<-matrix(0,nrow=length(dataset),ncol=5)

test_accu_summary<-as.data.table(test_accu_summary)
test_accu_summary$V1<-as.factor(test_accu_summary$V1)

for (i in 1:length(dataset)){
  test_accu_summary[i,1]=dataset[i]
  for (j in 1:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    data<-read.csv(tmp_file)
    
    test_accu_summary[i,j+1]=round(mean(data$test_accu),3)
  }
}
colnames(test_accu_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prototype","rank_full_model")

##
test_sd_summary<-matrix(0,nrow=length(dataset),ncol=5)
test_sd_summary<-as.data.table(test_sd_summary)
test_sd_summary$V1<-as.factor(test_sd_summary$V1)

for (i in 1:length(dataset)){
  test_sd_summary[i,1]=dataset[i]
  for (j in 1:length(models)){
    
    tmp=paste0(folder,m[j])
    setwd(tmp)
    
    #dataset[1]
    tmp_file<-paste0(dataset[i],file_name_extension[j])
    data<-read.csv(tmp_file)
    
    test_sd_summary[i,j+1]=round(sd(data$test_accu),3)
  }
}
colnames(test_sd_summary)<-c("dataset","rank_svm","rank_cg","rank_cg_prototype","rank_full_model")
##


#rank_cg_prot vs rank_svm----

t_test_p_values<-matrix(0,nrow=length(dataset),ncol=7)

t_test_p_values<-as.data.table(t_test_p_values)
t_test_p_values$V1<-as.factor(t_test_p_values$V1)
for (i in 1:length(dataset)){
  t_test_p_values[i,1]=dataset[i]
  counter=2
  for (j in (1:3)){
    #j=1
    #k=j+3
    for(k in ((j+1):4)){
    
      tmp=paste0(folder,m[j])
      setwd(tmp)
      tmp_file<-paste0(dataset[i],file_name_extension[j])
      data<-read.csv(tmp_file)
      data1_test_roc=data$test_roc
      #mean_test_roc_summary[i,j+1]=round(mean(data$test_roc),3)
      
      tmp=paste0(folder,m[k])
      setwd(tmp)
      tmp_file<-paste0(dataset[i],file_name_extension[k])
      data<-read.csv(tmp_file)
      data2_test_roc=data$test_roc
      
      t_test_result=t.test(data1_test_roc,data2_test_roc,paired=TRUE)
      t_test_p_values[i,counter]=round(t_test_result$p.value,4)
      counter=counter+1
    }
    
    
  }
}

tmp_names<-("dataset")
for(j in 1:3){
  for (k in (j+1):4){
    a=models[j]
    b=models[k]
    comb=paste0(a,"_vs_",b)
    tmp_names<-c(tmp_names,comb)
  }
}

colnames(t_test_p_values)<-tmp_names



## t test for accuracy

t_test_p_values_accu<-matrix(0,nrow=length(dataset),ncol=7)

t_test_p_values_accu<-as.data.table(t_test_p_values_accu)
t_test_p_values_accu$V1<-as.factor(t_test_p_values_accu$V1)
for (i in 1:length(dataset)){
  t_test_p_values_accu[i,1]=dataset[i]
  counter=2
  for (j in (1:3)){
    #j=1
    #k=j+3
    for(k in ((j+1):4)){
      
      tmp=paste0(folder,m[j])
      setwd(tmp)
      tmp_file<-paste0(dataset[i],file_name_extension[j])
      data<-read.csv(tmp_file)
      data1_test_roc=data$test_accu
      #mean_test_roc_summary[i,j+1]=round(mean(data$test_roc),3)
      
      tmp=paste0(folder,m[k])
      setwd(tmp)
      tmp_file<-paste0(dataset[i],file_name_extension[k])
      data<-read.csv(tmp_file)
      data2_test_roc=data$test_accu
      
      t_test_result=t.test(data1_test_roc,data2_test_roc,paired=TRUE)
      t_test_p_values_accu[i,counter]=round(t_test_result$p.value,4)
      counter=counter+1
    }
    
    
  }
}

tmp_names<-("dataset")
for(j in 1:3){
  for (k in (j+1):4){
    a=models[j]
    b=models[k]
    comb=paste0(a,"_vs_",b)
    tmp_names<-c(tmp_names,comb)
  }
}

colnames(t_test_p_values_accu)<-tmp_names







