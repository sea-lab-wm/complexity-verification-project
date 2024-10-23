"""
This script is for running the experiments with classification models.

Classification models: SVC, KNN, Logistic Regression, Random Forest, MLP

This requires the experiments.jsonl file to get the experiments, 
and the understandability_with_warnings.csv file to get the data.
After performing the GridSearchCV, it will find the best hyperparameters for each fold and train 
the model with the best hyperparameters.
"""

import sys
import logging

# from utils import configs
from utils import configs_TASK1_DS6 as configs
sys.path.append(configs.ROOT_PATH)


import pandas as pd
import csv
import json
import numpy as np

# import matplotlib.pyplot as plt

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.multiclass import unique_labels


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid

from sklearn.utils.parallel import Parallel, delayed
from joblib import parallel_backend ## train the models in parallel

from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import os


## keep the models determisitic
np.random.seed(configs.RANDOM_SEED)


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
    # "y_test_index": "",
    # "y_actual": "",
    # "y_pred": "",
    
    "tp_0": "-",
    "tn_0": "-",
    "fp_0": "-",
    "fn_0": "-",
    "support_0": "-",

    "tp_1": "-",
    "tn_1": "-",
    "fp_1": "-",
    "fn_1": "-",
    "support_1": "-",

    "tp_2": "-",
    "tn_2": "-",
    "fp_2": "-",
    "fn_2": "-",
    "support_2": "-",

    "tp_3": "-",
    "tn_3": "-",
    "fp_3": "-",
    "fn_3": "-",
    "support_3": "-",

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
    "auc_weighted": 0.0,

    "precision_macro": 0.0,
    "recall_macro": 0.0,
    "f1_macro": 0.0,
    "auc_macro": 0.0,

    "f1_0 > baseline": 0,
    "f1_1 > baseline": 0,
    "f1_2 > baseline": 0,
    "f1_3 > baseline": 0,

    "f1_0 = baseline": 0,
    "f1_1 = baseline": 0,
    "f1_2 = baseline": 0,
    "f1_3 = baseline": 0,

    "#_of_classes_improved_(f1>base)": 0,
    "#_of_classes_degraded_(f1<base)": 0,
    "#_of_classes_no_change_(f1=base)": 0,

    "F1_weighted > baseline": 0,
    "RI=(F1_weighted-baseline)/baseline": 0,

    "RI_F1_0": 0.0,
    "RI_F1_1": 0.0,
    "RI_F1_2": 0.0,
    "RI_F1_3": 0.0,
    "RI_weighted": 0.0,

    "Avg_RI_Improvement": 0.0, # average relative improvement for all classes that improved
    "Avg_RI_Degradation": 0.0, # average relative improvement for all classes that degraded

    "experiment": ""
}

def custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, cv, config, target):
    ## all the permutations of the hyperparameters ##
    candidate_params = list(ParameterGrid(param_grid))

    drop_duplicates = config["drop_duplicates"]
    best_hyperparameters_ = {} ## keep best hyperparameters for each fold and corresponding f1_weighted score

    # Process each fold in parallel
    results=Parallel(n_jobs=5)(delayed(process_fold)(fold, train_index, test_index, candidate_params, drop_duplicates, model_name, pipeline, X_train, y_train, target)
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
                 drop_duplicates, model_name, pipeline, X_train, y_train, target):
    
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
        results = evaluate(model, X_val_fold, y_val_fold, target)
  
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
    """
    train the model parallelly using Joblib
    """
    # Use a Joblib context manager to train the classifier in parallel
    # with parallel_backend('loky', n_jobs=8): 
    model.fit(X_train.values, y_train.values.ravel())
    return model


def getBestBaselineModel(target):
    ## best baseline = lazy0
    if target == "ABU":
        baseline_0_f1 = 0.90410958904
        baseline_1_f1 = 0.0
        baseline_f1_weighted = 0.7458904110

        baseline_2_f1 = 0
        baseline_3_f1 = 0

        class_0_weight = 0.825
        class_1_weight = 0.175
        
        class_2_weight = 0
        class_3_weight = 0
        
    ## best baseline = random(distribution)    
    elif target == "ABU50":
        baseline_0_f1 = 0.51136364
        baseline_1_f1 = 0.48863636
        baseline_f1_weighted = 0.500258264

        baseline_2_f1 = 0
        baseline_3_f1 = 0

        class_0_weight = 0.51
        class_1_weight = 0.49
        class_2_weight = 0
        class_3_weight = 0

    ## best baseline = random(distribution)
    elif target == "BD":
        baseline_0_f1 = 0.48409091
        baseline_1_f1 = 0.51590909
        baseline_f1_weighted = 0.50050620
        
        baseline_2_f1 = 0
        baseline_3_f1 = 0

        class_0_weight = 0.48
        class_1_weight = 0.52
        class_2_weight = 0
        class_3_weight = 0

    ## best baseline = lazy0
    elif target == "BD50":
        baseline_0_f1 = 0.887484197218711
        baseline_1_f1 = 0.0

        baseline_f1_weighted = 0.7080

        baseline_2_f1 = 0
        baseline_3_f1 = 0    

        class_0_weight = 0.8
        class_1_weight = 0.2
        class_2_weight = 0
        class_3_weight = 0

    ## best baseline = random(distribution)    
    elif target == "PBU":
        baseline_0_f1 = 0.30909091
        baseline_1_f1 = 0.69090909

        baseline_f1_weighted = 0.57289256

        baseline_2_f1 = 0
        baseline_3_f1 = 0

        class_0_weight = 0.31
        class_1_weight = 0.69
        class_2_weight = 0
        class_3_weight = 0

    ## best baseline = random(distribution)
    elif target == "AU":
        baseline_0_f1 = 0.34772727
        baseline_1_f1 = 0.16363636
        baseline_2_f1 = 0.31363636
        baseline_3_f1 = 0.1750

        baseline_f1_weighted = 0.27668388

        class_0_weight = 0.35
        class_1_weight = 0.16
        class_2_weight = 0.31
        class_3_weight = 0.18

    return baseline_0_f1, baseline_1_f1, baseline_2_f1, baseline_3_f1, class_0_weight, class_1_weight, class_2_weight, class_3_weight, baseline_f1_weighted        



def evaluate(model, X_test, y_test, target):
    ## Transform the test data to Scaler ##
    # X_test = model.named_steps.scaler.transform(X_test.values)
    X_test = X_test.values
    
    ## predict on the test split ##
    y_pred = model.predict(X_test)

    classifi_report = classification_report(y_test, y_pred, output_dict=True,  zero_division=0)
    prediction_report = multilabel_confusion_matrix (y_test, y_pred, labels=unique_labels(y_test))

    ## get baseline f1 scores and class weights ##
    baseline_f1_0, baseline_f1_1, baseline_f1_2, baseline_f1_3, class_0_weight, class_1_weight, class_2_weight, class_3_weight, baseline_f1_weighted = getBestBaselineModel(target)
    
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
        # y_pred_proba_1 = model.predict_proba(X_test)[:, 1]
        # y_pred_proba_0 = model.predict_proba(X_test)[:, 0]
        y_pred_proba_1 = np.zeros(len(y_test))
        y_pred_proba_0 = np.zeros(len(y_test))
        
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
        auc_macro = "-"

        support_2 = "-"
        support_3 = "-"

        f1_2_imporved_baseline = "-"
        f1_3_imporved_baseline = "-"
        f1_2_no_change_baseline = "-"
        f1_3_no_change_baseline = "-"

        RI_F1_2 = "-" ## CHANGE
        RI_F1_3 = "-" ## CHANGE

        
    else:
        ## For multi-class classification
        f1_1 = classifi_report["1"]["f1-score"]
        f1_0 = classifi_report["0"]["f1-score"]
        f1_2 = classifi_report["2"]["f1-score"]
        f1_3 = classifi_report["3"]["f1-score"]

        # auc_0 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[0]
        # auc_1 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[1]
        # auc_2 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[2]
        # auc_3 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[3]
        auc_0 = 0
        auc_1 = 0
        auc_2 = 0
        auc_3 = 0

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

        support_2 = classifi_report["2"]["support"]
        support_3 = classifi_report["3"]["support"]

        predict_test = np.zeros(len(X_test))
        # auc_weighted = roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovr", labels=unique_labels(y_test))
        auc_weighted = 0
        # auc_macro = roc_auc_score(y_test, model.predict_proba(X_test), average="macro", multi_class="ovr", labels=unique_labels(y_test))
        auc_macro = 0
        
        f1_2_imporved_baseline = 1 if f1_2 > baseline_f1_2 else 0
        f1_3_imporved_baseline = 1 if f1_3 > baseline_f1_3 else 0

        f1_2_no_change_baseline = 1 if f1_2 == baseline_f1_2 else 0
        f1_3_no_change_baseline = 1 if f1_3 == baseline_f1_3 else 0
        

        ## if f1_2 = 0 and baseline_f1_2 = 0, then RI_F1_2 = ""
        ## if f1_2 = 0 and baseline_f1_2 != 0, then RI_F1_2 = ""
        # if f1_2 != 0 and baseline_f1_2 = 0, then RI_F1_2 = ""
        # if f1_2 != 0 and baseline_f1_2 != 0, then (f1_2 - baseline_f1_2) / baseline_f1_2
        ## Implemenrt the above logic for all the classes
        RI_F1_2 = (f1_2 - baseline_f1_2) / baseline_f1_2 if (f1_2 != 0 and baseline_f1_2 != 0) else ""
        RI_F1_3 = (f1_3 - baseline_f1_3) / baseline_f1_3 if (f1_3 != 0 and baseline_f1_3 != 0) else ""
        


    support_0 = classifi_report["0"]["support"]
    support_1 = classifi_report["1"]["support"]

    ## compute the weighted scores ##
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=unique_labels(y_test), zero_division=0)
    precision_weighted = classifi_report["weighted avg"]["precision"]
    recall_weighted = classifi_report["weighted avg"]["recall"]

    ## compute the macro scores ##
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=unique_labels(y_test), zero_division=0)
    precision_macro = classifi_report["macro avg"]["precision"]
    recall_macro = classifi_report["macro avg"]["recall"]


    ## F1 improvement compared to baseline ##
    f1_0_imporved_baseline = 1 if f1_0 > baseline_f1_0 else 0
    f1_1_imporved_baseline = 1 if f1_1 > baseline_f1_1 else 0
    f1_0_no_change_baseline = 1 if f1_0 == baseline_f1_0 else 0
    f1_1_no_change_baseline = 1 if f1_1 == baseline_f1_1 else 0

    RI_F1_0 = (f1_0 - baseline_f1_0) / baseline_f1_0 if (f1_0 != 0 and baseline_f1_0 != 0) else ""
    RI_F1_1 = (f1_1 - baseline_f1_1) / baseline_f1_1 if (f1_1 != 0 and baseline_f1_1 != 0) else ""
    
    
    ## if target is binary
    if type_of_target(y_test) == "binary":
        num_of_classes_improved_to_baseline = f1_0_imporved_baseline + f1_1_imporved_baseline
        num_of_classes_detegraded_to_baseline = 2 - num_of_classes_improved_to_baseline
        num_of_classes_no_change_to_baseline = f1_0_no_change_baseline + f1_1_no_change_baseline
    else:    
        num_of_classes_improved_to_baseline = f1_0_imporved_baseline + f1_1_imporved_baseline + f1_2_imporved_baseline + f1_3_imporved_baseline
        num_of_classes_detegraded_to_baseline = 4 - num_of_classes_improved_to_baseline
        num_of_classes_no_change_to_baseline = f1_0_no_change_baseline + f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline

    FIXED_RI_F1_0 = 0 if (RI_F1_0 == "-" or RI_F1_0 =="") else RI_F1_0
    FIXED_RI_F1_1 = 0 if (RI_F1_1 == "-" or RI_F1_1=="") else RI_F1_1
    FIXED_RI_F1_2 = 0 if (RI_F1_2 == "-" or RI_F1_2 =="") else RI_F1_2
    FIXED_RI_F1_3 = 0 if (RI_F1_3 == "-" or RI_F1_3 == "") else RI_F1_3
    
    
    RI_weighted = FIXED_RI_F1_0 * class_0_weight + FIXED_RI_F1_1 * class_1_weight + FIXED_RI_F1_2 * class_2_weight + FIXED_RI_F1_3 * class_3_weight


    ## check the RI_F1_0, RI_F1_1, RI_F1_2, RI_F1_3 and filter out the classes that improved ##
    all_classes = [FIXED_RI_F1_0, FIXED_RI_F1_1, FIXED_RI_F1_2, FIXED_RI_F1_3]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    ## take the average of the improved classes ##
    if len(improved_classes) == 0:
        avg_RI_Improvement = ""
    else:
        avg_RI_Improvement = np.nanmean(improved_classes)
    if len(degraded_classes) == 0:
        avg_RI_Degradation = ""
    else:    
        avg_RI_Degradation = np.nanmean(degraded_classes)

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
    results_dict["precision_weighted"] = precision_weighted
    results_dict["recall_weighted"] = recall_weighted
    results_dict["auc_weighted"] = auc_weighted

    results_dict["f1_macro"] = f1_macro
    results_dict["precision_macro"] = precision_macro
    results_dict["recall_macro"] = recall_macro
    results_dict["auc_macro"] = auc_macro

    results_dict["y_test_index"] = y_test.index.values
    results_dict["y_actual"] = y_test.values
    results_dict["y_pred"] = y_pred

    results_dict["support_0"] = support_0
    results_dict["support_1"] = support_1
    results_dict["support_2"] = support_2
    results_dict["support_3"] = support_3
    
    ## this is for internal use ##
    # results_dict["y_predict_proba"] = model.predict_proba(X_test)
    results_dict["y_predict_proba"] = np.zeros(len(y_test))

    ## compare with the baseline ##
    results_dict["f1_0 > baseline"] = f1_0_imporved_baseline
    results_dict["f1_1 > baseline"] = f1_1_imporved_baseline
    results_dict["f1_2 > baseline"] = f1_2_imporved_baseline
    results_dict["f1_3 > baseline"] = f1_3_imporved_baseline

    results_dict["f1_0 = baseline"] = f1_0_no_change_baseline
    results_dict["f1_1 = baseline"] = f1_1_no_change_baseline
    results_dict["f1_2 = baseline"] = f1_2_no_change_baseline
    results_dict["f1_3 = baseline"] = f1_3_no_change_baseline


    results_dict["#_of_classes_improved_(f1>base)"] = num_of_classes_improved_to_baseline
    results_dict["#_of_classes_degraded_(f1<base)"] = num_of_classes_detegraded_to_baseline
    results_dict["#_of_classes_no_change_(f1=base)"] = num_of_classes_no_change_to_baseline

    results_dict["F1_weighted > baseline"] = 1 if f1_weighted > baseline_f1_weighted else 0
    results_dict["RI=(F1_weighted-baseline)/baseline"] = (f1_weighted - baseline_f1_weighted) / baseline_f1_weighted

    results_dict["RI_F1_0"] = RI_F1_0
    results_dict["RI_F1_1"] = RI_F1_1
    results_dict["RI_F1_2"] = RI_F1_2
    results_dict["RI_F1_3"] = RI_F1_3
    results_dict["RI_weighted"] = RI_weighted

    results_dict["Avg_RI_Improvement"] = avg_RI_Improvement
    results_dict["Avg_RI_Degradation"] = avg_RI_Degradation

    return results_dict



def aggregate_results(results, target, model_name, K, hyperparam):
    """
    aggregate the results from all the folds
    """
    overall_f1_0 = 0
    overall_f1_1 = 0
    overall_f1_2 = "-"
    overall_f1_3 = "-"

    overall_auc_0 = 0  
    overall_auc_1 = 0
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

    overall_f1_0 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_f1_1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]

    
    overall_precision_0 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_precision_1 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]
    overall_precision_2 = "-"
    overall_precision_3 = "-"

    overall_recall_0 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_recall_1 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]
    overall_recall_2 = "-"
    overall_recall_3 = "-"
    
    if type_of_target(y_true_all) != "binary":
        overall_f1_2 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
        overall_f1_3 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]

        overall_precision_2 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
        overall_precision_3 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]

        overall_recall_2 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
        overall_recall_3 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]

    overall_f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    overall_precsion_weighted = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    overall_recall_weighted = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

    overall_f1_macro = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_precision_macro = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_recall_macro = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    ## get baseline f1 scores and class weights ##
    baseline_f1_0, baseline_f1_1, baseline_f1_2, baseline_f1_3, class_0_weight, class_1_weight, class_2_weight, class_3_weight, baseline_f1_weighted = getBestBaselineModel(target)
    
    if type_of_target(y_true_all) == "binary":
        # y_predict_proba_all = np.array(y_predict_proba_all)
        ## convert y_predict_proba_all to a numpy array 2D for testing purposes
        y_predict_proba_all = np.array(y_predict_proba_all).reshape(-1, 2)

        # overall_auc_0 = roc_auc_score(y_true_all, y_predict_proba_all[:, 0])
        # overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all[:, 1])
        overall_auc_0 = 0
        overall_auc_1 = 0
        
        overall_auc_weighted = "-" ## not applicable for binary classification
        overall_auc_macro = "-" ## not applicable for binary classification
        f1_0_baseline_improved = 1 if overall_f1_0 > baseline_f1_0 else 0
        f1_1_baseline_improved = 1 if overall_f1_1 > baseline_f1_1 else 0

        f1_0_no_change_baseline = 1 if overall_f1_0 == baseline_f1_0 else 0
        f1_1_no_change_baseline = 1 if overall_f1_1 == baseline_f1_1 else 0
        
        f1_2_baseline_improved = "-"
        f1_3_baseline_improved = "-"

        f1_2_no_change_baseline = "-"
        f1_3_no_change_baseline = "-"

        RI_F1_0 = (overall_f1_0 - baseline_f1_0) / baseline_f1_0 if (overall_f1_0 != 0 and baseline_f1_0 != 0) else ""
        RI_F1_1 = (overall_f1_1 - baseline_f1_1) / baseline_f1_1 if (overall_f1_1 != 0 and baseline_f1_1 != 0) else ""

        
        RI_F1_2 = "-"
        RI_F1_3 = "-"


    else:
        # overall_auc_0 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[0]
        # overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[1]
        # overall_auc_2 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[2]
        # overall_auc_3 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[0,1,2,3])[3]
        # overall_auc_weighted = roc_auc_score(y_true_all, y_predict_proba_all, average="weighted", multi_class="ovr", labels=[0,1,2,3])
        # overall_auc_macro = roc_auc_score(y_true_all, y_predict_proba_all, average="macro", multi_class="ovr", labels=[0,1,2,3])

        overall_auc_0 = 0
        overall_auc_1 = 0
        overall_auc_2 = 0
        overall_auc_3 = 0
        overall_auc_weighted = "-"
        overall_auc_macro = "-"

        f1_0_baseline_improved = 1 if overall_f1_0 > baseline_f1_0 else 0
        f1_1_baseline_improved = 1 if overall_f1_1 > baseline_f1_1 else 0
        f1_2_baseline_improved = 1 if overall_f1_2 > baseline_f1_2 else 0
        f1_3_baseline_improved = 1 if overall_f1_3 > baseline_f1_3 else 0

        f1_0_no_change_baseline = 1 if overall_f1_0 == baseline_f1_0 else 0
        f1_1_no_change_baseline = 1 if overall_f1_1 == baseline_f1_1 else 0
        f1_2_no_change_baseline = 1 if overall_f1_2 == baseline_f1_2 else 0
        f1_3_no_change_baseline = 1 if overall_f1_3 == baseline_f1_3 else 0

        RI_F1_0 = (overall_f1_0 - baseline_f1_0) / baseline_f1_0 if (overall_f1_0 != 0 and baseline_f1_0 != 0) else ""
        RI_F1_1 = (overall_f1_1 - baseline_f1_1) / baseline_f1_1 if (overall_f1_1 != 0 and baseline_f1_1 != 0) else ""
        RI_F1_2 = (overall_f1_2 - baseline_f1_2) / baseline_f1_2 if (overall_f1_2 != 0 and baseline_f1_2 != 0) else ""
        RI_F1_3 = (overall_f1_3 - baseline_f1_3) / baseline_f1_3 if (overall_f1_3 != 0 and baseline_f1_3 != 0) else ""


    ## if target is binary
    if type_of_target(y_true_all) == "binary":
        num_of_classes_improved_to_baseline = f1_0_baseline_improved + f1_1_baseline_improved
        num_of_classes_degreded_to_baseline = 2 - num_of_classes_improved_to_baseline
        num_of_classes_no_change_to_baseline = f1_0_no_change_baseline + f1_1_no_change_baseline
    else:
        num_of_classes_improved_to_baseline = f1_0_baseline_improved + f1_1_baseline_improved + f1_2_baseline_improved + f1_3_baseline_improved
        num_of_classes_degreded_to_baseline = 4 - num_of_classes_improved_to_baseline
        num_of_classes_no_change_to_baseline = f1_0_no_change_baseline + f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline


    FIXED_RI_F1_0 = 0 if (RI_F1_0 == "" or RI_F1_0=="-") else RI_F1_0
    FIXED_RI_F1_1 = 0 if (RI_F1_1 == "" or RI_F1_1=="-") else RI_F1_1
    FIXED_RI_F1_2 = 0 if (RI_F1_2 == "" or RI_F1_2=="-") else RI_F1_2
    FIXED_RI_F1_3 = 0 if (RI_F1_3 == "" or RI_F1_3=="-") else RI_F1_3

    RI_weighted = FIXED_RI_F1_0 * class_0_weight + FIXED_RI_F1_1 * class_1_weight + FIXED_RI_F1_2 * class_2_weight + FIXED_RI_F1_3 * class_3_weight

    all_classes = [FIXED_RI_F1_0, FIXED_RI_F1_1, FIXED_RI_F1_2, FIXED_RI_F1_3]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    if len(improved_classes) == 0:
        avg_RI_Improvement = ""
    else:
        avg_RI_Improvement = np.nanmean(improved_classes)
    if len(degraded_classes) == 0:
        avg_RI_Degradation = ""
    else:    
        avg_RI_Degradation = np.nanmean(degraded_classes)

    num_instances = len(y_true_all)

    ## remove np.int64 from the array
    y_index_all = [int(num) for num in y_index_all]
    y_true_all = [int(num) for num in y_true_all]
    y_pred_all = [int(num) for num in y_pred_all]

    F1_weighted_improved_over_baseline = 1 if overall_f1_weighted > baseline_f1_weighted else 0
    RI_F1_weighted = (overall_f1_weighted - baseline_f1_weighted) / baseline_f1_weighted

    ### Draw Confusion Matrix ###
    # import matplotlib.pyplot as plt
    #     # precision = dict()
    #     # recall = dict()
    #     # for i in range(3):
    #     #     precision[i], recall[i], _ = precision_recall_curve(y_true_all,
    #     #                                                         y_predict_proba_all[:, i])
    #     #     plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    #     # plt.xlabel("recall")
    #     # plt.ylabel("precision")
    #     # plt.legend(loc="best")
    #     # plt.title("precision vs. recall curve")
    #     # plt.savefig(configs.ROOT_PATH + "/results/pr_curve/" + target + "_" + str(hyperparam) + ".png")    
        

    # import seaborn as sns

    # cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels(y_true_all))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels="auto", yticklabels="auto")
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # if not os.path.exists(configs.ROOT_PATH + "/results/confusion_matrix/ds_6/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/confusion_matrix/ds_6/" + target + "/" + model_name + "/")   
    # plt.savefig(configs.ROOT_PATH + "/results/confusion_matrix/ds_6/" + target + "/" + model_name +"/" + str(K) + "_" + target + "_" + str(hyperparam) + ".png")
    # plt.close()

    # ### SAVE THE PREDICTION RESULTS WITH INDEX and ACTUAL LABELS ###
    # ## if target is binary
    # if type_of_target(y_true_all) == "binary":
    #     results_df = pd.DataFrame(list(zip(y_index_all, y_true_all, y_pred_all, np.array([proba[0] for proba in y_predict_proba_all]), np.array([proba[1] for proba in y_predict_proba_all]), (np.where(np.array(y_true_all) == np.array(y_pred_all), 1, 0)))), columns=["index", "actual_label", "predicted_label", "proba_0", "proba_1", "correct_prediction"])
    # elif type_of_target(y_true_all) == "multiclass":
    #     results_df = pd.DataFrame(list(zip(y_index_all, y_true_all, y_pred_all, np.array([proba[0] for proba in y_predict_proba_all]), np.array([proba[1] for proba in y_predict_proba_all]), np.array([proba[2] for proba in y_predict_proba_all]), np.array([proba[3] for proba in y_predict_proba_all]), (np.where(np.array(y_true_all) == np.array(y_pred_all), 1, 0)))), columns=["index", "actual_label", "predicted_label", "proba_0", "proba_1", "proba_2", "proba_3", "correct_prediction"])
    # if not os.path.exists(configs.ROOT_PATH + "/results/predictions/ds_6/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/predictions/ds_6/" + target + "/" + model_name + "/")
    # results_df.to_csv(configs.ROOT_PATH + "/results/predictions/ds_6/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + ".csv", index=False)


    return overall_f1_0, overall_f1_1, overall_f1_2, overall_f1_3, overall_auc_0, overall_auc_1, overall_auc_2, overall_auc_3, overall_f1_weighted, overall_auc_weighted, overall_f1_macro, overall_auc_macro, overall_precision_0, overall_precision_1, overall_precision_2, overall_precision_3, overall_recall_0, overall_recall_1, overall_recall_2, overall_recall_3, overall_precsion_weighted, overall_recall_weighted, overall_precision_macro, overall_recall_macro, tp_0_all, tn_0_all, fp_0_all, fn_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, num_instances, y_index_all, y_true_all, y_pred_all, overall_f1_weighted, overall_f1_macro, overall_auc_weighted, overall_auc_macro, num_of_classes_improved_to_baseline, num_of_classes_degreded_to_baseline, num_of_classes_no_change_to_baseline ,RI_F1_0, RI_F1_1, RI_F1_2, RI_F1_3, avg_RI_Improvement, avg_RI_Degradation, RI_weighted, f1_0_baseline_improved, f1_1_baseline_improved, f1_2_baseline_improved, f1_3_baseline_improved, f1_0_no_change_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, F1_weighted_improved_over_baseline, RI_F1_weighted



def model_initialisation(model_name, parameters):
    LOGGER.info("Launching model: " + model_name + "...")
    if model_name == "logisticregression":
        ## parameters for grid search
        ## We picked the parameters based on the following resources as believe those are the most important parameters to tune:
        ## https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
        param_grid = {
            "C": [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1],
            "penalty": ["l2"], ## l2 is recommended since less sensitive to outliers
            "solver": ["lbfgs"], ## liblinear is recommended for small datasets and lbfgs for multi-class problems
            "max_iter": [8000],
            "multi_class": ["ovr"],
            "class_weight": ["balanced"],
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
            "n_neighbors": [3, 4, 5],
            "weights": ["uniform"], ## distance is given more weight to the closer points (weight = 1/distance)
            "metric": ["euclidean"], ## manhattan is the distance between two points measured along axes at right angles
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
            "n_estimators": [100, 150, 200, 300],
            "max_features": [None],
            "min_impurity_decrease": [0.001, 0.01 ],
            "max_depth": [5, 10, 15],
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
            "C": [1e-5, 1e-6, 1e-9],
            # "C": [1e-5],
            "kernel": ["linear"],
            # "tol": [1.0e-12, 1.0e-9, 1.0e-6],
            "tol": [1.0e-12, 1.0e-9, 1.0e-6],
            # "probability": [True],  # to compute the roc_auc score
            # "gamma": ["auto", "scale"],
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
            "learning_rate_init": [0.0001],
            "momentum":[0.2, 0.9],
            "activation": ["relu", "logistic"], ## logistic is sigmoid (Italian paper)
            "solver": ["adam", "sgd"],
            "max_iter":[2000],  # Adjust based on validation
            "early_stopping": [True],
            "random_state": [configs.RANDOM_SEED]
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
            "var_smoothing": [0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-12, 1e-15]
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = GaussianNB()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = GaussianNB(**parameters)

    return model, param_grid 

def dict_data_generator(model, iteration, hyperparameters, target, drop_duplicates, use_oversampling, dataset, K, fs_method, y_test_index, y_actual,y_pred, tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, tp_2, tn_2, fp_2, fn_2, tp_3, tn_3, fp_3, fn_3, f1_0, f1_1, f1_2, f1_3, support_0, support_1, support_2, support_3, auc_0, auc_1, auc_2, auc_3, precision_0, precision_1, precision_2, precision_3, recall_0, recall_1, recall_2, recall_3, n_instances, presion_weighted, recall_weighted, f1_weighted, f1_macro, auc_weighted, experiment, precision_macro, recall_macro, auc_macro, f1_0_imporved_baseline, f1_1_imporved_baseline, f1_2_imporved_baseline, f1_3_imporved_baseline, num_of_classes_improved_to_baseline, num_of_classes_detegraded_to_baseline, num_of_classes_no_change_to_baseline ,RI_F1_0, RI_F1_1, RI_F1_2, RI_F1_3, RI_weighted, avg_RI_Improvement, avg_RI_Degradation, f1_0_no_change_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, F1_weighted_improved_over_baseline, RI_F1_weighted): 
    
    csv_data_dict["model"] = model
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = hyperparameters
    csv_data_dict["target"] = target
    
    csv_data_dict["drop_duplicates"] = drop_duplicates
    csv_data_dict["use_oversampling"] = use_oversampling
    csv_data_dict["dataset"] = dataset
    
    csv_data_dict["K"] = K
    csv_data_dict["fs_method"] = fs_method
    
    # csv_data_dict["y_test_index"] = str(y_test_index).replace(", ", " ").replace("\n", " ")
    # csv_data_dict["y_actual"] = str(y_actual).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")
    # csv_data_dict["y_pred"] = str(y_pred).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")

    csv_data_dict["tp_0"] = tp_0
    csv_data_dict["tn_0"] = tn_0
    csv_data_dict["fp_0"] = fp_0
    csv_data_dict["fn_0"] = fn_0
    csv_data_dict["support_0"] = support_0

    csv_data_dict["tp_1"] = tp_1
    csv_data_dict["tn_1"] = tn_1
    csv_data_dict["fp_1"] = fp_1
    csv_data_dict["fn_1"] = fn_1
    csv_data_dict["support_1"] = support_1

    csv_data_dict["tp_2"] = tp_2
    csv_data_dict["tn_2"] = tn_2
    csv_data_dict["fp_2"] = fp_2
    csv_data_dict["fn_2"] = fn_2
    csv_data_dict["support_2"] = support_2

    csv_data_dict["tp_3"] = tp_3
    csv_data_dict["tn_3"] = tn_3
    csv_data_dict["fp_3"] = fp_3
    csv_data_dict["fn_3"] = fn_3
    csv_data_dict["support_3"] = support_3

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
    csv_data_dict["precision_weighted"] = presion_weighted
    csv_data_dict["recall_weighted"] = recall_weighted
    csv_data_dict["auc_weighted"] = auc_weighted

    csv_data_dict["f1_macro"] = f1_macro
    csv_data_dict["precision_macro"] = precision_macro
    csv_data_dict["recall_macro"] = recall_macro
    csv_data_dict["auc_macro"] = auc_macro

    csv_data_dict["f1_0 > baseline"] = f1_0_imporved_baseline
    csv_data_dict["f1_1 > baseline"] = f1_1_imporved_baseline
    csv_data_dict["f1_2 > baseline"] = f1_2_imporved_baseline
    csv_data_dict["f1_3 > baseline"] = f1_3_imporved_baseline

    csv_data_dict["f1_0 = baseline"] = f1_0_no_change_baseline
    csv_data_dict["f1_1 = baseline"] = f1_1_no_change_baseline
    csv_data_dict["f1_2 = baseline"] = f1_2_no_change_baseline
    csv_data_dict["f1_3 = baseline"] = f1_3_no_change_baseline

    csv_data_dict["#_of_classes_improved_(f1>base)"] = num_of_classes_improved_to_baseline
    csv_data_dict["#_of_classes_degraded_(f1<base)"] = num_of_classes_detegraded_to_baseline
    csv_data_dict["#_of_classes_no_change_(f1=base)"] = num_of_classes_no_change_to_baseline

    csv_data_dict["RI_F1_0"] = RI_F1_0
    csv_data_dict["RI_F1_1"] = RI_F1_1
    csv_data_dict["RI_F1_2"] = RI_F1_2
    csv_data_dict["RI_F1_3"] = RI_F1_3
    csv_data_dict["RI_weighted"] = RI_weighted

    csv_data_dict["F1_weighted > baseline"] = F1_weighted_improved_over_baseline
    csv_data_dict["RI=(F1_weighted-baseline)/baseline"] = RI_F1_weighted

    csv_data_dict["Avg_RI_Improvement"] = avg_RI_Improvement
    csv_data_dict["Avg_RI_Degradation"] = avg_RI_Degradation

    csv_data_dict["experiment"] = experiment
    

    return csv_data_dict



def main():
    complete_df = pd.read_csv(configs.ROOT_PATH + "/" + configs.DATA_PATH)
    
    ## write header
    with open(configs.ROOT_PATH + "/" + configs.OUTPUT_ML_PATH, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(configs.ROOT_PATH + "/" + configs.FILTERED_EXPERIMENTS) as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=configs.RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=configs.RANDOM_SEED)

    models = [  "randomForest_classifier", "logisticregression", "knn_classifier", "svc", "mlp_classifier", "bayes_network" ]

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
                pipeline = Pipeline(steps = [('scaler', RobustScaler()), (model_name, model)])
                # pipeline = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model)])

                ## CONFIG 1 ## - apply over sampling
                if use_oversampling: # if use RandomOverSampling
                        # ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        smote = SMOTE(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', RobustScaler()),   
                        # ('scaler', StandardScaler()), 
                        # ('ros', ros),
                        ('smo', smote),
                        (model_name, model)])

                ## CONFIG 2 ## - remove duplicates from training set
                config = {"drop_duplicates": drop_duplicates}
                
                best_params, best_score_ = custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, inner_cv, config, target)
                
                LOGGER.info("Best param searching for fold {} for code features...".format(fold))
                
                ## since we are using a set, we need to convert the dict to a hashable type
                best_hyperparams.add((frozenset(best_params)))

            # ## if there are no hyperparameters found use the default hyperparameters
            # if len(best_hyperparams) == 0:
            #     best_hyperparams.add(frozenset({}))

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
                    pipeline = Pipeline(steps = [('scaler', RobustScaler()), (model_name, model)])
                    # pipeline = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model)])
                    if experiment['use_oversampling']:
                        # ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        smo = SMOTE(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', RobustScaler()), 
                        # ('scaler', StandardScaler()),   
                        # ('ros', ros),
                        ('smo', smo),
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
                    fold_results_ = evaluate(model, X_test, y_test, target)

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
                        support_0=fold_results_["support_0"],

                        tp_1=fold_results_["tp_1"],
                        tn_1=fold_results_["tn_1"],
                        fp_1=fold_results_["fp_1"], 
                        fn_1=fold_results_["fn_1"], 
                        support_1=fold_results_["support_1"],

                        tp_2=fold_results_["tp_2"], 
                        tn_2=fold_results_["tn_2"],
                        fp_2=fold_results_["fp_2"], 
                        fn_2=fold_results_["fn_2"],
                        support_2=fold_results_["support_2"],

                        tp_3=fold_results_["tp_3"],
                        tn_3=fold_results_["tn_3"],
                        fp_3=fold_results_["fp_3"],
                        fn_3=fold_results_["fn_3"],
                        support_3=fold_results_["support_3"],

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
                        auc_weighted=fold_results_["auc_weighted"], 

                        precision_macro=fold_results_["precision_macro"],
                        f1_macro=fold_results_["f1_macro"],
                        recall_macro=fold_results_["recall_macro"],
                        auc_macro=fold_results_["auc_macro"],

                        f1_0_imporved_baseline=fold_results_["f1_0 > baseline"],
                        f1_1_imporved_baseline=fold_results_["f1_1 > baseline"],
                        f1_2_imporved_baseline=fold_results_["f1_2 > baseline"],
                        f1_3_imporved_baseline=fold_results_["f1_3 > baseline"],

                        f1_0_no_change_baseline=fold_results_["f1_0 = baseline"],
                        f1_1_no_change_baseline=fold_results_["f1_1 = baseline"],
                        f1_2_no_change_baseline=fold_results_["f1_2 = baseline"],
                        f1_3_no_change_baseline=fold_results_["f1_3 = baseline"],

                        num_of_classes_improved_to_baseline=fold_results_["#_of_classes_improved_(f1>base)"],
                        num_of_classes_detegraded_to_baseline=fold_results_["#_of_classes_degraded_(f1<base)"],
                        num_of_classes_no_change_to_baseline=fold_results_["#_of_classes_no_change_(f1=base)"],

                        RI_F1_0=fold_results_["RI_F1_0"],
                        RI_F1_1=fold_results_["RI_F1_1"],
                        RI_F1_2=fold_results_["RI_F1_2"],
                        RI_F1_3=fold_results_["RI_F1_3"],
                        RI_weighted=fold_results_["RI_weighted"],

                        F1_weighted_improved_over_baseline=fold_results_["F1_weighted > baseline"],
                        RI_F1_weighted=fold_results_["RI=(F1_weighted-baseline)/baseline"],

                        avg_RI_Improvement=fold_results_["Avg_RI_Improvement"],
                        avg_RI_Degradation=fold_results_["Avg_RI_Degradation"],
                        
                        experiment=experiment["exp_id"]) 
                    

                    dict_to_csv(configs.ROOT_PATH + "/" + configs.OUTPUT_ML_PATH, csv_data_dict1)

                ## aggregate the results from all the folds
                overall_f1_0, overall_f1_1, overall_f1_2, overall_f1_3, overall_auc_0, overall_auc_1, overall_auc_2, overall_auc_3, overall_f1_weighted, overall_auc_weighted, overall_f1_macro, overall_auc_macro, overall_precision_0, overall_precision_1, overall_precision_2, overall_precision_3, overall_recall_0, overall_recall_1, overall_recall_2, overall_recall_3, overall_precsion_weighted, overall_recall_weighted, overall_precision_macro, overall_recall_macro, tp_0_all, tn_0_all, fp_0_all, fn_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, num_instances, y_index_all, y_true_all, y_pred_all, overall_f1_weighted, overall_f1_macro, overall_auc_weighted, overall_auc_macro, num_of_classes_improved_to_baseline, num_of_classes_degreded_to_baseline, num_of_classes_no_change_to_baseline ,RI_F1_0, RI_F1_1, RI_F1_2, RI_F1_3, avg_RI_Improvement, avg_RI_Degradation, RI_weighted, f1_0_baseline_improved, f1_1_baseline_improved, f1_2_baseline_improved, f1_3_baseline_improved, f1_0_no_change_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, F1_weighted_improved_over_baseline, RI_F1_weighted = aggregate_results(all_fold_results, target, model_name, experiment["K %"], best_hyper_params)  
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
                    support_0=tp_0_all + fn_0_all,

                    tp_1=tp_1_all,
                    tn_1=tn_1_all,
                    fp_1=fp_1_all, 
                    fn_1=fn_1_all, 
                    support_1=tp_1_all + fn_1_all,

                    tp_2=tp_2_all, 
                    tn_2=tn_2_all,
                    fp_2=fp_2_all, 
                    fn_2=fn_2_all,
                    support_2=tp_2_all + fn_2_all,

                    tp_3=tp_3_all,
                    tn_3=tn_3_all,
                    fp_3=fp_3_all,
                    fn_3=fn_3_all,
                    support_3=tp_3_all + fn_3_all,

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
                    auc_weighted=overall_auc_weighted, 

                    f1_macro=overall_f1_macro,
                    precision_macro=overall_precision_macro,
                    recall_macro=overall_recall_macro,
                    auc_macro=overall_auc_macro,

                    f1_0_imporved_baseline=f1_0_baseline_improved,
                    f1_1_imporved_baseline=f1_1_baseline_improved,
                    f1_2_imporved_baseline=f1_2_baseline_improved,
                    f1_3_imporved_baseline=f1_3_baseline_improved,

                    f1_0_no_change_baseline=f1_0_no_change_baseline,
                    f1_1_no_change_baseline=f1_1_no_change_baseline,
                    f1_2_no_change_baseline=f1_2_no_change_baseline,
                    f1_3_no_change_baseline=f1_3_no_change_baseline,

                    num_of_classes_improved_to_baseline=num_of_classes_improved_to_baseline,
                    num_of_classes_detegraded_to_baseline=num_of_classes_degreded_to_baseline,
                    num_of_classes_no_change_to_baseline=num_of_classes_no_change_to_baseline,

                    F1_weighted_improved_over_baseline=F1_weighted_improved_over_baseline,
                    RI_F1_weighted=RI_F1_weighted,

                    RI_F1_0=RI_F1_0,
                    RI_F1_1=RI_F1_1,
                    RI_F1_2=RI_F1_2,
                    RI_F1_3=RI_F1_3,
                    RI_weighted=RI_weighted,

                    avg_RI_Improvement=avg_RI_Improvement,
                    avg_RI_Degradation=avg_RI_Degradation,

                    experiment=experiment["exp_id"]
                    )
                    
                dict_to_csv(configs.ROOT_PATH + "/" + configs.OUTPUT_ML_PATH, csv_data_dict2)

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

if __name__ == "__main__":
    main()