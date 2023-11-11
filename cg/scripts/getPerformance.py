from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,f1_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
import time
import scipy

def getPerformance(real_train_outcomes, real_test_outcomes, predicted_train_outcomes,predicted_test_outcomes,tr_scores,te_scores):
	roc_train = roc_auc_score(real_train_outcomes, tr_scores)
	roc_test = roc_auc_score(real_test_outcomes, te_scores)

	acc_train = accuracy_score(real_train_outcomes, predicted_train_outcomes)
	acc_test = accuracy_score(real_test_outcomes, predicted_test_outcomes)

	sensitivity_score_train = sensitivity_score(real_train_outcomes, predicted_train_outcomes)
	sensitivity_score_test = sensitivity_score(real_test_outcomes, predicted_test_outcomes)

	specificity_score_train = specificity_score(real_train_outcomes, predicted_train_outcomes)
	specificity_score_test = specificity_score(real_test_outcomes, predicted_test_outcomes)

	geometric_mean_score_train = geometric_mean_score(real_train_outcomes, predicted_train_outcomes)
	geometric_mean_score_test = geometric_mean_score(real_test_outcomes, predicted_test_outcomes)

	precision_train = precision_score(real_train_outcomes, predicted_train_outcomes)
	precision_test = precision_score(real_test_outcomes, predicted_test_outcomes)

	f1_score_train = f1_score(real_train_outcomes, predicted_train_outcomes)
	f1_score_test = f1_score(real_test_outcomes, predicted_test_outcomes)

	return [roc_train, acc_train, sensitivity_score_train, specificity_score_train, geometric_mean_score_train, precision_train, f1_score_train,\
		 roc_test, acc_test, sensitivity_score_test, specificity_score_test, geometric_mean_score_test, precision_test, f1_score_test]