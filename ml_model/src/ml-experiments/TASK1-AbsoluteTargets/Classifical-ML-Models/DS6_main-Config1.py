import sys
import logging
sys.path.append('/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/')

from utils import configs

import pandas as pd
import csv
import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.multiclass import unique_labels


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from sklearn.utils.parallel import Parallel, delayed

from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, roc_curve

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import os

df = pd.read_csv(configs.ROOT_PATH + "/" + configs.DATA_PATH)

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')




csv_data_dict = {
    "model": "",
    "iteration": "",
    "hyperparameters": "",
    "target": "",
    "drop_duplicates": "",
    "use_oversampling": False,
    "dataset": "",
    "K": 0,
    "fs_method": "",
    "y_test_index": "",
    "y_actual": "",
    "y_pred": "",
    "tp_0": "-",
    "tn_0": "-",
    "fp_0": "-",
    "fn_0": "-",
    "tp_1": "-",
    "tn_1": "-",
    "fp_1": "-",
    "fn_1": "-",
    "tp_2": "-",
    "tn_2": "-",
    "fp_2": "-",
    "fn_2": "-",
    "tp_3": "-",
    "tn_3": "-",
    "fp_3": "-",
    "fn_3": "-",
    "n_instances": 0,
    "precision_0": "-",
    "precision_1": "-",
    "precision_2": "-",
    "precision_3": "-",
    "recall_0": "-",
    "recall_1": "-",
    "recall_2": "-",
    "recall_3": "-",
    "f1_0": "-",
    "f1_1": "-",
    "f1_2": "-",
    "f1_3": "-",
    "auc_0": "-",
    "auc_1": "-",
    "auc_2": "-",
    "auc_3": "-",
    "precision_weighted": 0.0,
    "recall_weighted": 0.0,
    "f1_weighted": 0.0,
    "f1_macro": 0.0,
    "auc_weighted": 0.0,
    "experiment": ""
}

def custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, cv, config):
    ## all the permutations of the hyperparameters ##
    candidate_params = list(ParameterGrid(param_grid))

    drop_duplicates = config["drop_duplicates"]
    best_hyperparameters_ = {} ## keep best hyperparameters for each fold and corresponding f1_weighted score

    # Process each fold in parallel
    results=Parallel(n_jobs=-1)(delayed(process_fold)(fold, train_index, test_index, candidate_params, drop_duplicates, model_name, pipeline, X_train, y_train)
                                  for fold, (train_index, test_index) in enumerate(cv.split(X_train, y_train)))

    # Collect results for each fold
    for (fold, best_param_, best_score_) in results:
        best_hyperparameters_[fold] = {"hyperparameters": best_param_ , "f1_weighted": best_score_}

    best_score_ = best_hyperparameters_[0]["f1_weighted"]
    best_params_ = best_hyperparameters_[0]["hyperparameters"]

    # Determine the best hyperparameters across all folds
    for fold, best_hyperparameters in best_hyperparameters_.items():
        score_ = best_hyperparameters["f1_weighted"]
        if best_score_ < score_:
            best_score_ = score_
            best_params_ = best_hyperparameters["hyperparameters"]

    return best_params_, best_score_     
        
   
       
def process_fold(fold, train_index, test_index, candidate_params, 
                 drop_duplicates, model_name, pipeline, X_train, y_train):
    
    ## best hyper params for the current fold
    hyper_param_dict= {}

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    ## drop duplicates from the training folds only ##
    if drop_duplicates:
        combined_df = pd.concat([X_train_fold, y_train_fold], axis=1)

        ## Identify duplicate rows
        duplicates_mask = combined_df.duplicated()

        ## Remove duplicates from X_train_c and y_train_c
        X_train_fold = X_train_fold[~duplicates_mask]
        y_train_fold = y_train_fold[~duplicates_mask]


    for (cand_idx, parameters) in enumerate(candidate_params):
        
        ## train the model with the current hyperparameters
        model = train(pipeline.set_params(**parameters), X_train_fold, y_train_fold)
        results = evaluate(model, X_val_fold, y_val_fold)
  
        ## remove the model name from the best hyperparameters keys
        hyperparam = ({key.replace(model_name + "__", ""):value for key, value in parameters.items()}) 

        ## convert the dict to a hashable type
        hyper_param_dict[frozenset(hyperparam.items())] = results


    ## get the best hyperparameters for the current fold ##
    best_score_ = hyper_param_dict[list(hyper_param_dict.keys())[0]]["f1_weighted"]
    best_param_ = list(hyper_param_dict.keys())[0]

    for hyper_param in hyper_param_dict:
        f1_weighted_score = hyper_param_dict[hyper_param]["f1_weighted"]
        if best_score_ < f1_weighted_score:
            best_param_ = hyper_param
            best_score_ = f1_weighted_score

    return fold, best_param_, best_score_



def train(model, X_train, y_train):
    ## train the model on the train split
    model.fit(X_train.values, y_train.values.ravel())
    return model


def evaluate(model, X_test, y_test):
    ## Transform the test data to Scaler ##
    X_test = model.named_steps.scaler.transform(X_test.values)
    
    ## predict on the test split ##
    y_pred = model.predict(X_test)

    classifi_report = classification_report(y_test, y_pred, output_dict=True)
    prediction_report = multilabel_confusion_matrix (y_test, y_pred, labels=unique_labels(y_test))

    
    #### ONLY FOR BINARY CLASSIFICATION ####
    if type_of_target(y_test) == "binary":
        f1_1 = classifi_report["1"]["f1-score"]
        f1_0 = classifi_report["0"]["f1-score"]
        precision_0 = classifi_report["0"]["precision"]
        precision_1 = classifi_report["1"]["precision"]
        recall_0 = classifi_report["0"]["recall"]
        recall_1 = classifi_report["1"]["recall"]
        
        ## Predict probabilities ## 
        ## source : https://scikit-learn.org/stable/modules/model_evaluation.html#binary-case
        ## In the case of providing the probability estimates, the probability of the class 
        ## with the “greater label” should be provided. The “greater label” corresponds to classifier.classes_[1] and 
        ## thus classifier.predict_proba(X)[:, 1].
        y_pred_proba_1 = model.predict_proba(X_test)[:, 1]
        y_pred_proba_0 = model.predict_proba(X_test)[:, 0]
        
        ## It does not make sense to calculate AUC for binary classification
        ## source: https://stackoverflow.com/questions/42059805/how-should-i-get-the-auc-for-the-negative-class
        auc_0 = roc_auc_score(y_test, y_pred_proba_0) 
        auc_1 = roc_auc_score(y_test, y_pred_proba_1)

        f1_2 = "-"
        f1_3 = "-"
        precision_2 = "-"
        precision_3 = "-"
        recall_2 = "-"
        recall_3 = "-"
        auc_2 = "-"
        auc_3 = "-"

        tp_0 = prediction_report[0][1][1]
        tn_0 = prediction_report[0][0][0]
        fp_0 = prediction_report[0][0][1]
        fn_0 = prediction_report[0][1][0]

        tp_1 = prediction_report[1][1][1]
        tn_1 = prediction_report[1][0][0]
        fp_1 = prediction_report[1][0][1]
        fn_1 = prediction_report[1][1][0]

        tp_2 = "-"
        tn_2 = "-"
        fp_2 = "-"
        fn_2 = "-"

        tp_3 = "-"
        tn_3 = "-"
        fp_3 = "-"
        fn_3 = "-"

        auc_weighted = "-"
    else:
        ## For multi-class classification
        f1_1 = classifi_report["1"]["f1-score"]
        f1_0 = classifi_report["0"]["f1-score"]
        f1_2 = classifi_report["2"]["f1-score"]
        f1_3 = classifi_report["3"]["f1-score"]

        auc_0 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[0]
        auc_1 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[1]
        auc_2 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[2]
        auc_3 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[3]

        precision_0 = classifi_report["0"]["precision"]
        precision_1 = classifi_report["1"]["precision"]
        precision_2 = classifi_report["2"]["precision"]
        precision_3 = classifi_report["3"]["precision"]

        recall_0 = classifi_report["0"]["recall"]
        recall_1 = classifi_report["1"]["recall"]
        recall_2 = classifi_report["2"]["recall"]
        recall_3 = classifi_report["3"]["recall"]

        tp_0 = prediction_report[0][1][1]
        tn_0 = prediction_report[0][0][0]
        fp_0 = prediction_report[0][0][1]
        fn_0 = prediction_report[0][1][0]

        tp_1 = prediction_report[1][1][1]
        tn_1 = prediction_report[1][0][0]
        fp_1 = prediction_report[1][0][1]
        fn_1 = prediction_report[1][1][0]

        tp_2 = prediction_report[2][1][1]
        tn_2 = prediction_report[2][0][0]
        fp_2 = prediction_report[2][0][1]
        fn_2 = prediction_report[2][1][0]

        tp_3 = prediction_report[3][1][1]
        tn_3 = prediction_report[3][0][0]
        fp_3 = prediction_report[3][0][1]
        fn_3 = prediction_report[3][1][0]

        auc_weighted = roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovr", labels=unique_labels(y_test))
    
    ## compute the weighted scores ##
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=unique_labels(y_test))
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=unique_labels(y_test))
    precision_weighted = classifi_report["weighted avg"]["precision"]
    recall_weighted = classifi_report["weighted avg"]["recall"]

    
    results_dict = {}

    ## For multi-class classification
    results_dict["tp_0"] = tp_0
    results_dict["tn_0"] = tn_0
    results_dict["fp_0"] = fp_0
    results_dict["fn_0"] = fn_0

    results_dict["tp_1"] = tp_1
    results_dict["tn_1"] = tn_1
    results_dict["fp_1"] = fp_1
    results_dict["fn_1"] = fn_1

    results_dict["tp_2"] = tp_2
    results_dict["tn_2"] = tn_2
    results_dict["fp_2"] = fp_2
    results_dict["fn_2"] = fn_2

    results_dict["tp_3"] = tp_3
    results_dict["tn_3"] = tn_3
    results_dict["fp_3"] = fp_3
    results_dict["fn_3"] = fn_3

    results_dict["f1_0"] = f1_0
    results_dict["f1_1"] = f1_1
    results_dict["f1_2"] = f1_2
    results_dict["f1_3"] = f1_3

    results_dict["precision_0"] = precision_0
    results_dict["precision_1"] = precision_1
    results_dict["precision_2"] = precision_2
    results_dict["precision_3"] = precision_3

    results_dict["recall_0"] = recall_0
    results_dict["recall_1"] = recall_1
    results_dict["recall_2"] = recall_2
    results_dict["recall_3"] = recall_3

    results_dict["auc_0"] = auc_0
    results_dict["auc_1"] = auc_1
    results_dict["auc_2"] = auc_2
    results_dict["auc_3"] = auc_3

    results_dict["n_instances"] = len(y_test)
    results_dict["f1_weighted"] = f1_weighted
    results_dict["f1_macro"] = f1_macro
    results_dict["precision_weighted"] = precision_weighted
    results_dict["recall_weighted"] = recall_weighted
    results_dict["auc_weighted"] = auc_weighted


    results_dict["y_test_index"] = y_test.index.values
    results_dict["y_actual"] = y_test.values
    results_dict["y_pred"] = y_pred

    ## this is for internal use ##
    results_dict["y_predict_proba"] = model.predict_proba(X_test)

    return results_dict


def generate_y_scores(y_true):
    """                 
    eg:
                    actual
                    0	                        |     1
                ----------------------------------------------------------------
                %	x = n_class0/444 = 80.63%   |    y = n_class1/444 = 19.37%
                #	n_class0 = 358	            |    n_class1 = 86
                ----------------------------------------------------------------
    prediction  0	n_class0*(x) = 288.66 	    |    n_class1*(x) = 69.34 
                1	** n_class0*(y) = 69.34 ** 	|    ** n_class1*(y) = 16.66 **
    
    
    ## Note: We selected threshold = 0.5 
    ## CLASS 0 ##
    proportion_class0_above_05 = int(69.34) = 69  # 19.37% of class 0 probability scores should be > 0.5
    proportion_class0_below_05 = 358 - 69 = 289  # 80.63% of class 0 probability scores should be < 0.5

    ## CLASS 1 ##
    proportion_class1_above_05 = int(16.66) = 16  # 19.37% of class 1 probability scores should be > 0.5
    proportion_class1_below_05 = 86 - 16 = 70  # 80.63% of class 1 probability scores should be < 0.5
    """

    # Initialize the y_scores array
    y_scores = np.zeros_like([item for sublist in y_true for item in sublist], dtype=float)

    ## Proportion calculations ##
    n_class1 = np.unique_counts(y_true)[1][1]
    n_class0 = np.unique_counts(y_true)[1][0]

    if type_of_target(y_true) == "binary":

        proportion_class0_above_05 = int(n_class1/(n_class1 + n_class0) * n_class0) 
        proportion_class0_below_05 = n_class0 - proportion_class0_above_05 

        proportion_class1_above_05 = int(n_class1/(n_class1 + n_class0) * n_class1)  # 19.37% of class 1 should be > 0.5
        proportion_class1_below_05 = n_class1 - proportion_class1_above_05  # 80.63% of class 1 should be < 0.5

        # Assign scores for class 0 #
        # pick random values between including 0 and excludng 0.5
        y_scores[:proportion_class0_below_05] = np.random.uniform(0., 0., int(proportion_class0_below_05))  # Below 0.5
        y_scores[proportion_class0_below_05:n_class0] = np.random.uniform(1., 1., int(proportion_class0_above_05))  # Above 0.5

        # Assign scores for class 1
        y_scores[n_class0 : (n_class0 + proportion_class1_below_05)] = np.random.uniform(0., 0., int(proportion_class1_below_05))  # Below 0.5
        y_scores[(n_class0 + proportion_class1_below_05):] = np.random.uniform(1., 1., int(proportion_class1_above_05))  # Above 0.5

    else: 
        n_class2 = np.unique_counts(y_true)[1][2]
        n_class3 = np.unique_counts(y_true)[1][3]

        proportion_class0_above_05 = int(n_class0/(n_class1 + n_class0 + n_class2 + n_class3) * n_class0) 
        proportion_class0_below_05 = n_class0 - proportion_class0_above_05 

        proportion_class1_above_05 = int(n_class1/(n_class1 + n_class0 + n_class2 + n_class3) * n_class1)  # 19.37% of class 1 should be > 0.5
        proportion_class1_below_05 = n_class1 - proportion_class1_above_05

        proportion_class2_above_05 = int(n_class2/(n_class1 + n_class0 + n_class2 + n_class3) * n_class2)
        proportion_class2_below_05 = n_class2 - proportion_class2_above_05

        proportion_class3_above_05 = int(n_class3/(n_class1 + n_class0 + n_class2 + n_class3) * n_class3)
        proportion_class3_below_05 = n_class3 - proportion_class3_above_05

        y_scores[:proportion_class0_below_05] = np.random.uniform(0., 0., int(proportion_class0_below_05))  # Below 0.5
        y_scores[proportion_class0_below_05:n_class0] = np.random.uniform(1, 1., int(proportion_class0_above_05))  # Above 0.5

        y_scores[n_class0 : (n_class0 + proportion_class1_below_05)] = np.random.uniform(0., 0., int(proportion_class1_below_05))  # Below 0.5
        y_scores[(n_class0 + proportion_class1_below_05):(n_class0 + n_class1)] = np.random.uniform(1., 1., int(proportion_class1_above_05))

        y_scores[(n_class0 + n_class1) : (n_class0 + n_class1 + proportion_class2_below_05)] = np.random.uniform(0., 0., int(proportion_class2_below_05))  # Below 0.5 
        y_scores[(n_class0 + n_class1 + proportion_class2_below_05):(n_class0 + n_class1 + n_class2)] = np.random.uniform(1, 1., int(proportion_class2_above_05))

        y_scores[(n_class0 + n_class1 + n_class2) : (n_class0 + n_class1 + n_class2 + proportion_class3_below_05)] = np.random.uniform(0., 0., int(proportion_class3_below_05))  # Below 0.5
        y_scores[(n_class0 + n_class1 + n_class2 + proportion_class3_below_05):] = np.random.uniform(1., 1., int(proportion_class3_above_05))



    return y_scores

# def create_results_dict_per_fold(y_test, y_pred):
#     confusion_matrix_ = multilabel_confusion_matrix(y_test, y_pred, labels=unique_labels(y_test))
#     per_fold_results = {}
#     ## count the occuraces of the label types. #1 and #0
#     results = {}
#     for i, label in enumerate(unique_labels(y_test)):
#         results[label] = confusion_matrix_[i] ## tn, fp, fn, tp
#     classes = list(results.keys())
#     num_class_instances = {}
#     per_class_predictions = {}
#     for label in classes:    
#         if str(label) not in num_class_instances:
#             num_class_instances[str(label)] = 0
#             #number of instances
#             num_class_instances[str(label)] += (results[label][1][0] + results[label][1][1]) ## fn + tp
#     for label in classes:
#         per_class_predictions[label] = [0, 0, 0, 0] ## tn, fp, fn, tp
#         ## tn, fp, fn, tp
#         tn = results[label][0][0]
#         fp = results[label][0][1]
#         fn = results[label][1][0]
#         tp = results[label][1][1]
#         per_class_predictions[label]=[tn, fp, fn, tp]
#     per_fold_results["total tp"] = sum([per_class_predictions[label][3] for label in classes])
#     per_fold_results["total tn"] = sum([per_class_predictions[label][0] for label in classes])
#     per_fold_results["total fp"] = sum([per_class_predictions[label][1] for label in classes])
#     per_fold_results["total fn"] = sum([per_class_predictions[label][2] for label in classes])
#     per_fold_results["per_class_predictions"] = per_class_predictions
#     per_fold_results["num_of_instances"] = num_class_instances
#     return per_fold_results 

# def generate_random_y_pred(y_true_all, consider_distribution):
    
#     ## unique values
#     unique_values = np.unique(y_true_all)

#     for val in unique_values:
#         ## variable name with str(val)
#         exec(f"num_{val}_all = y_true_all.count({val})")
        

#     # Calculate the number of times to guess 1 and 0
#     if consider_distribution:
#         ## Consider distribution
#         for val in unique_values:
#             exec(f"num_guess_{val} = round(num_{val}_all * (num_{val}_all / len(y_true_all)))")
#     else:
#         ## Uniform distribution
#         for val in unique_values:
#             exec(f"num_guess_{val} = round(num_{val}_all * (1/len(unique_values)))")
    

#     # Initialize y_pred_all
#     y_pred_all = []

#     ## save the above exectued variables in a dictionary and use it to implement the logic
#     dict_vars = locals()

#     # Initialize the number of guesses for 1s and 0s
#     num_guess_ones = dict_vars['num_guess_1']
#     num_guess_zeros = dict_vars['num_guess_0']

#     num_guess_twos = 0
#     num_guess_threes = 0

#     ## if dict_vars['num_guess_2'] and dict_vars['num_guess_3'] there, then use them as well
#     if 'num_guess_2' in dict_vars:
#         num_guess_twos = dict_vars['num_guess_2']
#     if 'num_guess_3' in dict_vars:
#         num_guess_threes = dict_vars['num_guess_3']

#     for y in y_true_all:
#         if y == 1 and num_guess_ones > 0:
#             y_pred_all.append(1)
#             num_guess_ones -= 1
#         elif y == 0 and num_guess_zeros > 0:
#             y_pred_all.append(0)
#             num_guess_zeros -= 1
#         elif y == 2 and num_guess_twos > 0:
#             y_pred_all.append(2)
#             num_guess_twos -= 1
#         elif y == 3 and num_guess_threes > 0:
#             y_pred_all.append(3)
#             num_guess_threes -= 1
#         else:
#             ## if binary classification
#             if unique_values.shape[0] == 2:
#                 y_pred_all.append(0 if y == 1 else 1)
#             ## if multi-class classification    
#             if unique_values.shape[0] == 3:
#                 if y==1:
#                     y_pred_all.append(np.random.choice([0,2]))
#                 elif y==0:
#                     y_pred_all.append(np.random.choice([1,2]))
#                 elif y==2:
#                     y_pred_all.append(np.random.choice([0,1]))
#             if unique_values.shape[0] == 4:
#                 if y==1:
#                     y_pred_all.append(np.random.choice([0,2,3]))
#                 elif y==0:
#                     y_pred_all.append(np.random.choice([1,2,3]))
#                 elif y==2:
#                     y_pred_all.append(np.random.choice([0,1,3]))
#                 elif y==3:
#                     y_pred_all.append(np.random.choice([0,1,2]))
    
#     return y_pred_all


def aggregate_results(results, target, model_name, hyperparameter, experiment):
    """
    aggregate the results from all the folds
    """
    overall_f1_0 = 0
    overall_f1_1 = 0

    overall_auc_1 = 0

    overall_f1_2 = "-"
    overall_f1_3 = "-"
    overall_auc_2 = "-"
    overall_auc_3 = "-"

    overall_f1_weighted = 0
    overall_auc_weighted = 0

    tp_0_all = 0
    tn_0_all = 0
    fp_0_all = 0
    fn_0_all = 0

    tp_1_all = 0
    tn_1_all = 0
    fp_1_all = 0
    fn_1_all = 0

    tp_2_all = "-"
    tn_2_all = "-"
    fp_2_all = "-"
    fn_2_all = "-"

    tp_3_all = "-"
    tn_3_all = "-"
    fp_3_all = "-"
    fn_3_all = "-"

    y_index_all =[]
    y_pred_all = []
    y_true_all = []
    y_predict_proba_all = []

    for key, value in results.items():
        y_true_all.extend(value["y_actual"])

        
    if type_of_target(y_true_all) != "binary":
        tp_2_all = 0
        tn_2_all = 0
        fp_2_all = 0
        fn_2_all = 0

        tp_3_all = 0
        tn_3_all = 0
        fp_3_all = 0
        fn_3_all = 0

        overall_f1_2 = 0
        overall_f1_3 = 0

        overall_auc_2 = 0
        overall_auc_3 = 0
    
   

    for key, value in results.items():
        y_index_all.extend(value["y_test_index"])
        y_pred_all.extend(value["y_pred"])

        ## this is for internal use
        y_predict_proba_all.extend(value["y_predict_proba"])
        
        tp_0_all += value["tp_0"]
        tn_0_all += value["tn_0"]
        fp_0_all += value["fp_0"]
        fn_0_all += value["fn_0"]

        tp_1_all += value["tp_1"]
        tn_1_all += value["tn_1"]
        fp_1_all += value["fp_1"]
        fn_1_all += value["fn_1"]

        if type_of_target(y_true_all) != "binary":
            tp_2_all += value["tp_2"]
            tn_2_all += value["tn_2"]
            fp_2_all += value["fp_2"]
            fn_2_all += value["fn_2"]

            tp_3_all += value["tp_3"]
            tn_3_all += value["tn_3"]
            fp_3_all += value["fp_3"]
            fn_3_all += value["fn_3"]

    overall_f1_0 = f1_score(y_true_all, y_pred_all, average=None)[0]
    overall_f1_1 = f1_score(y_true_all, y_pred_all, average=None)[1]

    
    overall_precision_0 = precision_score(y_true_all, y_pred_all, average=None)[0]
    overall_precision_1 = precision_score(y_true_all, y_pred_all, average=None)[1]
    overall_precision_2 = "-"
    overall_precision_3 = "-"

    overall_recall_0 = recall_score(y_true_all, y_pred_all, average=None)[0]
    overall_recall_1 = recall_score(y_true_all, y_pred_all, average=None)[1]
    overall_recall_2 = "-"
    overall_recall_3 = "-"
    
    if type_of_target(y_true_all) != "binary":
        overall_f1_2 = f1_score(y_true_all, y_pred_all, average=None)[2]
        overall_f1_3 = f1_score(y_true_all, y_pred_all, average=None)[3]

        overall_precision_2 = precision_score(y_true_all, y_pred_all, average=None)[2]
        overall_precision_3 = precision_score(y_true_all, y_pred_all, average=None)[3]

        overall_recall_2 = recall_score(y_true_all, y_pred_all, average=None)[2]
        overall_recall_3 = recall_score(y_true_all, y_pred_all, average=None)[3]

    overall_f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted")
    overall_f1_macro = f1_score(y_true_all, y_pred_all, average="macro")
    overall_precsion_weighted = precision_score(y_true_all, y_pred_all, average="weighted")
    overall_recall_weighted = recall_score(y_true_all, y_pred_all, average="weighted")
    
    if type_of_target(y_true_all) == "binary":
        y_predict_proba_all = np.array(y_predict_proba_all)
        overall_auc_0 = roc_auc_score(y_true_all, y_predict_proba_all[:, 0])
        overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all[:, 1])
        overall_auc_weighted = "-" ## not applicable for binary classification
    else:
        overall_auc_0 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[0]
        overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[1]
        overall_auc_2 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[2]
        overall_auc_3 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[3]
        overall_auc_weighted = roc_auc_score(y_true_all, y_predict_proba_all, average="weighted", multi_class="ovr", labels=[0,1,2,3])



    ## draw the roc curve
    if type_of_target(y_true_all) == "binary":
        plt.figure()
        y_predict_proba_all = np.array(y_predict_proba_all)
        ## for positive class
        fpr, tpr, _ = roc_curve(y_true_all, y_predict_proba_all[:, 1])
        roc_auc = roc_auc_score(y_true_all, y_predict_proba_all[:, 1])
                
        ## for negative class
        ## source: https://stackoverflow.com/questions/42059805/how-should-i-get-the-auc-for-the-negative-class
        fpr_0, tpr_0, _ = roc_curve(y_true_all, y_predict_proba_all[:, 0])        
        roc_auc_0 = roc_auc_score(y_true_all, y_predict_proba_all[:, 0])
       
        ## Plot ROC curve for class 1 the model ## 
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve class1 (area = %0.2f)' % roc_auc)

        ## Plot ROC curve for class 0 the model ##
        plt.plot(fpr_0, tpr_0, color='green', lw=2, label='ROC curve class0 (area = %0.2f)' % roc_auc_0)
        

        y_pred_prob_naive = generate_y_scores(y_true_all)
        fpr_naive, tpr_naive, _ = roc_curve(y_true_all, y_pred_prob_naive)
        roc_auc_naive = roc_auc_score(y_true_all, y_pred_prob_naive)

        ## Plot the random guesser curve ##
        plt.plot(fpr_naive, tpr_naive, color='red', lw=2, linestyle='--', label='Naive/Random guesser (area = %0.2f)' % roc_auc_naive)
        
    
    else :
        ## plot the roc curve for multi-class classification
        plt.figure()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_predict_proba_all1 = np.array(y_predict_proba_all)
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(y_true_all, y_predict_proba_all1[:, i], pos_label=i)
            roc_auc[i] = roc_auc_score(y_true_all, y_predict_proba_all1, average=None, multi_class="ovr", labels=[0,1,2,3])[i]
            plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve class %d (area = %0.2f)' % (i, roc_auc[i]))
            
            ## Consider class i as the positive class and the rest as the negative class. Assign 1 for the positive class and 0 for the rest
            # y_true_all_transformed = [1 if x == i else 0 for x in y_true_all]
            # fpr_naive, tpr_naive, _ = roc_curve(y_true_all_transformed, y_predict_proba_all1[:, i])
            # roc_auc_naive = roc_auc_score(y_true_all_transformed, y_predict_proba_all1[:, i])
            # plt.plot(fpr_naive, tpr_naive, lw=2, linestyle='--', label='Naive/Random guesser class %d (area = %0.2f)' % (i, roc_auc_naive))
        

    # Labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])   
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()


        
    ## if experiment directory does not exist create it
    if not os.path.exists(configs.ROOT_PATH + "/NewExperiments/results/ROC_AUC_CURVES/" + experiment["exp_id"] + "/" + model_name):
        os.makedirs(configs.ROOT_PATH + "/NewExperiments/results/ROC_AUC_CURVES/" + experiment["exp_id"]  + "/" + model_name)
    plt.savefig(configs.ROOT_PATH + "/NewExperiments/results/ROC_AUC_CURVES/" + experiment["exp_id"]  + "/" + model_name + "/" + target + "_" + str(hyperparameter) + "_roc_curve.png")

    num_instances = len(y_true_all)

    ## remove np.int64 from the array
    y_index_all = [int(num) for num in y_index_all]
    y_true_all = [int(num) for num in y_true_all]
    y_pred_all = [int(num) for num in y_pred_all]

    return overall_f1_macro, overall_f1_weighted, overall_auc_weighted, overall_precsion_weighted, overall_recall_weighted, overall_f1_0, overall_f1_1, overall_f1_2, overall_f1_3, overall_auc_0, overall_auc_1, overall_auc_2, overall_auc_3, overall_precision_0, overall_precision_1, overall_precision_2, overall_precision_3, overall_recall_0, overall_recall_1, overall_recall_2, overall_recall_3, tp_0_all, tn_0_all, fp_0_all, fn_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, num_instances, y_index_all, y_true_all, y_pred_all 




def model_initialisation(model_name, parameters):
    LOGGER.info("Launching model: " + model_name + "...")

    if model_name == "logisticregression":
        ## parameters for grid search
        ## We picked the parameters based on the following resources as believe those are the most important parameters to tune:
        ## https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
        param_grid = {
            # "C": [0.01, 0.1, 1],
            # "penalty": ["l2"], ## l2 is recommended since less sensitive to outliers
            # "solver": ["liblinear", "lbfgs"], ## liblinear is recommended for small datasets and lbfgs for multi-class problems
            # "max_iter": [8000],
            # "multi_class": ["auto"], ## ovr supports binary. multinomial is for multi-class. auto will pick based on the data
            "random_state": [configs.RANDOM_SEED]
        }
        ## Pipeline requires the model name before the parameters
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()}
        model = LogisticRegression() 
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = LogisticRegression(**parameters)
               
    elif model_name == "knn_classifier":
        ## https://www.kaggle.com/code/arunimsamudra/k-nn-with-hyperparameter-tuning?scriptVersionId=32640489&cellId=42
        param_grid = {
            "n_neighbors": [1, 3, 5, 7, 9, 10],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = KNeighborsClassifier()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = KNeighborsClassifier(**parameters)
        
    elif model_name == "randomForest_classifier":
        ## https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_features": ['sqrt', 'log2'],
            "min_impurity_decrease": [0.001, 0.01 ],
            "max_depth": [None, 10],
            # "max_depth": [30, 50],
            # "bootstrap": [True],
            # "min_samples_split": [10, 15],
            # "min_samples_leaf": [4, 5],
            "random_state": [configs.RANDOM_SEED]
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = RandomForestClassifier()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = RandomForestClassifier(**parameters)

    elif model_name == "svc":
        ## https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb#:~:text=The%20most%20critical%20hyperparameters%20for%20SVM%20are%20kernel%20%2C%20C%20%2C%20and,to%20make%20it%20linearly%20separable.
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["sigmoid"],
            "tol": [1.0e-12, 1.0e-3],
            "coef0":[1.0],
            "probability": [True], ## to compute the roc_auc score,
            # "gamma":["auto", "scale"],
            "random_state": [configs.RANDOM_SEED]
        }

        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = SVC()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = SVC(**parameters) 

    elif model_name == "mlp_classifier":
        ## https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
        param_grid = {
            "hidden_layer_sizes": [7, (50,)], ## Single hidden layer with 7 nodes (Italian paper)
            "learning_rate_init": [0.3],
            "momentum":[0.2, 0.9],
            "activation": ["logistic", "relu"], ## logistic is sigmoid (Italian paper)
            "solver": ["lbfgs", "adam", "sgd"],
            "max_iter":[500],  # Adjust based on validation
            "random_state": [configs.RANDOM_SEED],
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = MLPClassifier()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = MLPClassifier(**parameters)

    elif model_name == "bayes_network":
        ## https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-naive-bayes#3
        param_grid = {
            # "var_smoothing": [1e-9, 1e-12]
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = GaussianNB()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = GaussianNB(**parameters)
    
    return model, param_grid

def dict_data_generator(model, iteration, hyperparameters, target, drop_duplicates, use_oversampling, dataset, K, fs_method, y_test_index, y_actual,y_pred, tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, tp_2, tn_2, fp_2, fn_2, tp_3, tn_3, fp_3, fn_3, f1_0, f1_1, f1_2, f1_3, auc_0, auc_1, auc_2, auc_3, precision_0, precision_1, precision_2, precision_3, recall_0, recall_1, recall_2, recall_3, n_instances, presion_weighted, recall_weighted, f1_weighted, f1_macro, auc_weighted, experiment):
    csv_data_dict["model"] = model
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = hyperparameters
    csv_data_dict["target"] = target
    
    csv_data_dict["drop_duplicates"] = drop_duplicates
    csv_data_dict["use_oversampling"] = use_oversampling
    csv_data_dict["dataset"] = dataset
    
    csv_data_dict["K"] = K
    csv_data_dict["fs_method"] = fs_method
    
    csv_data_dict["y_test_index"] = str(y_test_index).replace(", ", " ").replace("\n", " ")
    csv_data_dict["y_actual"] = str(y_actual).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")
    csv_data_dict["y_pred"] = str(y_pred).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")

    csv_data_dict["tp_0"] = tp_0
    csv_data_dict["tn_0"] = tn_0
    csv_data_dict["fp_0"] = fp_0
    csv_data_dict["fn_0"] = fn_0

    csv_data_dict["tp_1"] = tp_1
    csv_data_dict["tn_1"] = tn_1
    csv_data_dict["fp_1"] = fp_1
    csv_data_dict["fn_1"] = fn_1

    csv_data_dict["tp_2"] = tp_2
    csv_data_dict["tn_2"] = tn_2
    csv_data_dict["fp_2"] = fp_2
    csv_data_dict["fn_2"] = fn_2

    csv_data_dict["tp_3"] = tp_3
    csv_data_dict["tn_3"] = tn_3
    csv_data_dict["fp_3"] = fp_3
    csv_data_dict["fn_3"] = fn_3

    csv_data_dict["f1_0"] = f1_0
    csv_data_dict["f1_1"] = f1_1
    csv_data_dict["f1_2"] = f1_2
    csv_data_dict["f1_3"] = f1_3

    csv_data_dict["auc_0"] = auc_0
    csv_data_dict["auc_1"] = auc_1
    csv_data_dict["auc_2"] = auc_2
    csv_data_dict["auc_3"] = auc_3

    csv_data_dict["precision_0"] = precision_0
    csv_data_dict["precision_1"] = precision_1
    csv_data_dict["precision_2"] = precision_2
    csv_data_dict["precision_3"] = precision_3

    csv_data_dict["recall_0"] = recall_0
    csv_data_dict["recall_1"] = recall_1
    csv_data_dict["recall_2"] = recall_2
    csv_data_dict["recall_3"] = recall_3

    csv_data_dict["n_instances"] = n_instances

    csv_data_dict["f1_weighted"] = f1_weighted
    csv_data_dict["f1_macro"] = f1_macro
    
    csv_data_dict["precision_weighted"] = presion_weighted
    csv_data_dict["recall_weighted"] = recall_weighted
    
    csv_data_dict["auc_weighted"] = auc_weighted

    csv_data_dict["experiment"] = experiment
    

    return csv_data_dict

def get_best_hyperparameters(best_param_score_dict):

    best_params_ = {}
    best_score_ = 0
    for best_params, best_score in best_param_score_dict.items():
        if best_score_ < best_score:
            best_score_ = best_score
            best_params_ = best_params

    return best_params_


def main():
    complete_df = pd.read_csv(configs.ROOT_PATH + "/" + configs.DATA_PATH)
    output_file = "classification_results_RQ1_CONFIG1.csv"
    
    ## write header
    with open(configs.ROOT_PATH + "/NewExperiments/results/" + output_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(configs.ROOT_PATH + "/NewExperiments/featureselection/experiments_RQ1_new_filtered.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=configs.RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=configs.RANDOM_SEED)

    models = ["svc", "randomForest_classifier", "knn_classifier", "bayes_network", "mlp_classifier", "logisticregression"]

    for model_name in models:
        
        for experiment in experiments:
            
            ######################
            ### CONFIGURATIONS ###
            ######################
            drop_duplicates = experiment["drop_duplicates"]
            use_oversampling = experiment["use_oversampling"]

            ###################
            ## INPUTS/OUTPUTS##
            ###################
            target = experiment["target"]
            feature_set = experiment["features"]

            feature_X = complete_df[feature_set]
            target_y = complete_df[target]

            all_fold_results = {}

            ######################
            ## Best Hyperparams ##
            ######################
            best_hyperparams = set()

            ########################################
            ## BEST HYPERPARAMETERS FOR EACH FOLD ##
            ########################################
            for (fold, (train_index,test_index))  in (enumerate(outer_cv.split(complete_df[feature_set], target_y))):
                ###################
                ## Code features ##
                ###################
                
                ## split the Code dataset into train and test
                X_train, X_test, y_train, y_test = (
                    pd.DataFrame(feature_X).iloc[train_index],
                    pd.DataFrame(feature_X).iloc[test_index],
                    pd.DataFrame(target_y).iloc[train_index],
                    pd.DataFrame(target_y).iloc[test_index]
                )

                model, param_grid = model_initialisation(model_name, parameters="")
                pipeline = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model)])
                
                ## CONFIG 1 ## - apply over sampling
                if use_oversampling: # if use RandomOverSampling
                        ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('ros', ros),
                        (model_name, model)])

                ## CONFIG 2 ## - remove duplicates from training set
                config = {"drop_duplicates": drop_duplicates}
                
                best_params, best_score_ = custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, inner_cv, config)
                
                LOGGER.info("Best param searching for fold {} for code features...".format(fold))
                
                ## since we are using a set, we need to convert the dict to a hashable type
                best_hyperparams.add((frozenset(best_params)))



            ##############################################
            ## Train and Test with best hyperparameters ##
            ##############################################
            ## Split the complete set = features + warning features in to 10 folds using outer_cv
            for best_hyper_params in best_hyperparams:
                
                for (fold, (train_index,test_index)) in enumerate(outer_cv.split(complete_df[feature_set], target_y)):
                    
                    ###################
                    ## Code features ##
                    ###################
                    ## split the Code dataset into train and test
                    X_train, X_test, y_train, y_test = (
                        pd.DataFrame(feature_X).iloc[train_index],
                        pd.DataFrame(feature_X).iloc[test_index],
                        pd.DataFrame(target_y).iloc[train_index],
                        pd.DataFrame(target_y).iloc[test_index]
                    )
                

                    model, _ = model_initialisation(model_name, parameters=dict(best_hyper_params))
                    pipeline = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model)])
                    
                    if experiment['use_oversampling']:
                        ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('ros', ros),
                        (model_name, model)])

                    config ={"drop_duplicates": drop_duplicates}

                    ## drop duplicates from the complete set
                    # Combine X_train_c and y_train_c
                    if drop_duplicates:
                        combined_df = pd.concat([X_train, y_train], axis=1)

                        # Identify duplicate rows
                        duplicates_mask = combined_df.duplicated()

                        # Remove duplicates from X_train_c and y_train_c
                        X_train = X_train[~duplicates_mask]
                        y_train = y_train[~duplicates_mask]

                    model = train(pipeline, X_train, y_train)
                    fold_results_ = evaluate(model, X_test, y_test)

                    all_fold_results[fold] = fold_results_

                    ## Per fold results 
                    csv_data_dict1 = dict_data_generator(
                        model=model_name, 
                        iteration=fold, 
                        hyperparameters=best_hyper_params, 
                        target=target, 
                        drop_duplicates=drop_duplicates, 
                        use_oversampling=use_oversampling, 
                        dataset=experiment["dataset"], 
                        K=experiment["K %"], 
                        fs_method=experiment["feature_selection_method"], 
                        y_test_index=fold_results_["y_test_index"], 
                        y_actual=fold_results_["y_actual"],
                        y_pred=fold_results_["y_pred"], 
                        tp_0=fold_results_["tp_0"], 
                        tn_0=fold_results_["tn_0"], 
                        fp_0=fold_results_["fp_0"],
                        fn_0=fold_results_["fn_0"],
                        tp_1=fold_results_["tp_1"],
                        tn_1=fold_results_["tn_1"],
                        fp_1=fold_results_["fp_1"], 
                        fn_1=fold_results_["fn_1"], 
                        tp_2=fold_results_["tp_2"], 
                        tn_2=fold_results_["tn_2"],
                        fp_2=fold_results_["fp_2"], 
                        fn_2=fold_results_["fn_2"],
                        tp_3=fold_results_["tp_3"],
                        tn_3=fold_results_["tn_3"],
                        fp_3=fold_results_["fp_3"],
                        fn_3=fold_results_["fn_3"],
                        f1_0=fold_results_["f1_0"], 
                        f1_1=fold_results_["f1_1"], 
                        f1_2=fold_results_["f1_2"], 
                        f1_3=fold_results_["f1_3"],
                        auc_0=fold_results_["auc_0"],
                        auc_1=fold_results_["auc_1"],
                        auc_2=fold_results_["auc_2"],
                        auc_3=fold_results_["auc_3"],
                        precision_0=fold_results_["precision_0"],
                        precision_1=fold_results_["precision_1"],
                        precision_2=fold_results_["precision_2"],
                        precision_3=fold_results_["precision_3"],
                        recall_0=fold_results_["recall_0"],
                        recall_1=fold_results_["recall_1"],
                        recall_2=fold_results_["recall_2"],
                        recall_3=fold_results_["recall_3"],
                        n_instances=fold_results_["n_instances"],
                        presion_weighted=fold_results_["precision_weighted"],
                        recall_weighted=fold_results_["recall_weighted"],
                        f1_weighted=fold_results_["f1_weighted"], 
                        f1_macro=fold_results_["f1_macro"],
                        auc_weighted=fold_results_["auc_weighted"], 
                        experiment=experiment["exp_id"]) 
                    

                    dict_to_csv(configs.ROOT_PATH + "/NewExperiments/results/" + output_file, csv_data_dict1)

                ## aggregate the results from all the folds
                overall_f1_macro, overall_f1_weighted, overall_auc_weighted, overall_precsion_weighted, overall_recall_weighted, overall_f1_0, overall_f1_1, overall_f1_2, overall_f1_3, overall_auc_0, overall_auc_1, overall_auc_2, overall_auc_3, overall_precision_0, overall_precision_1, overall_precision_2, overall_precision_3, overall_recall_0, overall_recall_1, overall_recall_2, overall_recall_3, tp_0_all, tn_0_all, fp_0_all, fn_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, num_instances, y_index_all, y_true_all, y_pred_all = aggregate_results(all_fold_results, target, model_name, best_hyper_params, experiment)
                csv_data_dict2 = dict_data_generator(
                    model=model_name, 
                    iteration="Overall", 
                    hyperparameters=best_hyper_params, 
                    target=target, 
                    drop_duplicates=drop_duplicates, 
                    use_oversampling=use_oversampling, 
                    dataset=experiment["dataset"], 
                    K=experiment["K %"], 
                    fs_method=experiment["feature_selection_method"], 
                    y_test_index=y_index_all, 
                    y_actual=y_true_all,
                    y_pred=y_pred_all, 
                    tp_0=tp_0_all, 
                    tn_0=tn_0_all, 
                    fp_0=fp_0_all,
                    fn_0=fn_0_all,
                    tp_1=tp_1_all,
                    tn_1=tn_1_all,
                    fp_1=fp_1_all, 
                    fn_1=fn_1_all, 
                    tp_2=tp_2_all, 
                    tn_2=tn_2_all,
                    fp_2=fp_2_all, 
                    fn_2=fn_2_all,
                    tp_3=tp_3_all,
                    tn_3=tn_3_all,
                    fp_3=fp_3_all,
                    fn_3=fn_3_all,
                    f1_0=overall_f1_0, 
                    f1_1=overall_f1_1, 
                    f1_2=overall_f1_2, 
                    f1_3=overall_f1_3,
                    auc_0=overall_auc_0,
                    auc_1=overall_auc_1,
                    auc_2=overall_auc_2,
                    auc_3=overall_auc_3,
                    precision_0=overall_precision_0,
                    precision_1=overall_precision_1,
                    precision_2=overall_precision_2,
                    precision_3=overall_precision_3,
                    recall_0=overall_recall_0,
                    recall_1=overall_recall_1,
                    recall_2=overall_recall_2,
                    recall_3=overall_recall_3,
                    n_instances=num_instances,
                    presion_weighted=overall_precsion_weighted,
                    recall_weighted=overall_recall_weighted,
                    f1_weighted=overall_f1_weighted, 
                    f1_macro=overall_f1_macro,
                    auc_weighted=overall_auc_weighted, 
                    experiment=experiment["exp_id"])
                    
                dict_to_csv(configs.ROOT_PATH + "/NewExperiments/results/" + output_file, csv_data_dict2)

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

if __name__ == "__main__":
    main()