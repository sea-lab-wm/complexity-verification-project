import sys
import logging
from utils import configs_DS3
sys.path.append(configs_DS3.ROOT_PATH)


import pandas as pd
import csv
import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.multiclass import unique_labels


from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid

from sklearn.utils.parallel import Parallel, delayed

from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, roc_curve

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

import os


## keep the models determisitic
np.random.seed(configs_DS3.RANDOM_SEED)

df = pd.read_csv(configs_DS3.ROOT_PATH + "/" + configs_DS3.DATA_PATH)

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
    
    "tp_4": "-",
    "tn_4": "-",
    "fp_4": "-",
    "fn_4": "-",
    "support_4": "-",
    
    "tp_5": "-",
    "tn_5": "-",
    "fp_5": "-",
    "fn_5": "-",
    "support_5": "-",

    "n_instances": 0,

    "precision_1": "-",
    "precision_2": "-",
    "precision_3": "-",
    "precision_4": "-",
    "precision_5": "-",

    "recall_1": "-",
    "recall_2": "-",
    "recall_3": "-",
    "recall_4": "-",
    "recall_5": "-",

    "f1_1": "-",
    "f1_2": "-",
    "f1_3": "-",
    "f1_4": "-",
    "f1_5": "-",

    "auc_1": "-",
    "auc_2": "-",
    "auc_3": "-",
    "auc_4": "-",
    "auc_5": "-",

    "precision_weighted": 0.0,
    "recall_weighted": 0.0,
    "f1_weighted": 0.0,
    "auc_weighted": 0.0,

    "precision_macro": 0.0,
    "recall_macro": 0.0,
    "f1_macro": 0.0,
    "auc_macro": 0.0,

    "f1_1 > baseline": 0,
    "f1_2 > baseline": 0,
    "f1_3 > baseline": 0,
    "f1_4 > baseline": 0,
    "f1_5 > baseline": 0,

    "f1_1 = baseline": 0,
    "f1_2 = baseline": 0,
    "f1_3 = baseline": 0,
    "f1_4 = baseline": 0,
    "f1_5 = baseline": 0,

    "#_of_classes_improved_(f1>base)": 0,
    "#_of_classes_degraded_(f1<base)": 0,
    "#_of_classes_no_change_(f1=base)": 0,

    "RI_F1_1": 0.0,
    "RI_F1_2": 0.0,
    "RI_F1_3": 0.0,
    "RI_F1_4": 0.0,
    "RI_F1_5": 0.0,
    "RI_F1_weighted": 0.0,

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
    results=Parallel(n_jobs=-1)(delayed(process_fold)(fold, train_index, test_index, candidate_params, drop_duplicates, model_name, pipeline, X_train, y_train, target)
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
        combined_df = pd.concat([X_train_fold, y_train_fold], axis=1) ## axis=1 for columns

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
    ## train the model on the train split
    model.fit(X_train.values, y_train.values.ravel())
    return model

def getBestBaselineModel(target):
    if target == "readability_level":
        baseline_1_precision = 0.0735
        baseline_1_recall = 0.0735
        baseline_1_f1 = 0.0735

        baseline_2_precision = 0.2050
        baseline_2_recall = 0.2050
        baseline_2_f1 = 0.2050

        baseline_3_precision = 0.2678
        baseline_3_recall = 0.2678
        baseline_3_f1 = 0.2678

        baseline_4_precision = 0.2719
        baseline_4_recall = 0.2719
        baseline_4_f1 = 0.2719

        baseline_5_precision = 0.1818
        baseline_5_recall = 0.1818
        baseline_5_f1 = 0.1818

        baseline_f1_weighted = 0.2261

        class_1_weight = 0.07
        class_2_weight = 0.21
        class_3_weight = 0.27
        class_4_weight = 0.27
        class_5_weight = 0.18
        

    return baseline_1_f1, baseline_2_f1, baseline_3_f1, baseline_4_f1, baseline_5_f1, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight       


def evaluate(model, X_test, y_test, target):
    ## Transform the test data to Scaler ##
    X_test = model.named_steps.scaler.transform(X_test.values)
    
    ## predict on the test split ##
    y_pred = model.predict(X_test)

    classifi_report = classification_report(y_test, y_pred, output_dict=True)
    prediction_report = multilabel_confusion_matrix (y_test, y_pred, labels=unique_labels(y_test))

    ## get baseline f1 scores and class weights ##
    baseline_1_f1, baseline_2_f1, baseline_3_f1, baseline_4_f1, baseline_5_f1, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight = getBestBaselineModel(target)

    ## For multi-class classification
    f1_1 = classifi_report["1"]["f1-score"]
    f1_2 = classifi_report["2"]["f1-score"]
    f1_3 = classifi_report["3"]["f1-score"]
    f1_4 = classifi_report["4"]["f1-score"]
    f1_5 = classifi_report["5"]["f1-score"]

    auc_1 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[0]
    auc_2 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[1]
    auc_3 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[2]
    auc_4 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[3]
    auc_5 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[4]

    precision_1 = classifi_report["1"]["precision"]
    precision_2 = classifi_report["2"]["precision"]
    precision_3 = classifi_report["3"]["precision"]
    precision_4 = classifi_report["4"]["precision"]
    precision_5 = classifi_report["5"]["precision"]

    recall_1 = classifi_report["1"]["recall"]
    recall_2 = classifi_report["2"]["recall"]
    recall_3 = classifi_report["3"]["recall"]
    recall_4 = classifi_report["4"]["recall"]
    recall_5 = classifi_report["5"]["recall"]

    tp_1 = prediction_report[0][1][1]
    tn_1 = prediction_report[0][0][0]
    fp_1 = prediction_report[0][0][1]
    fn_1 = prediction_report[0][1][0]

    tp_2 = prediction_report[1][1][1]
    tn_2 = prediction_report[1][0][0]
    fp_2 = prediction_report[1][0][1]
    fn_2 = prediction_report[1][1][0]

    tp_3 = prediction_report[2][1][1]
    tn_3 = prediction_report[2][0][0]
    fp_3 = prediction_report[2][0][1]
    fn_3 = prediction_report[2][1][0]

    tp_4 = prediction_report[3][1][1]
    tn_4 = prediction_report[3][0][0]
    fp_4 = prediction_report[3][0][1]
    fn_4 = prediction_report[3][1][0]

    tp_5 = prediction_report[4][1][1]
    tn_5 = prediction_report[4][0][0]
    fp_5 = prediction_report[4][0][1]
    fn_5 = prediction_report[4][1][0]

    ## compute the weighted scores ##
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=unique_labels(y_test))
    precision_weighted = classifi_report["weighted avg"]["precision"]
    recall_weighted = classifi_report["weighted avg"]["recall"]
    auc_weighted = roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovr", labels=unique_labels(y_test))
    
    ## compute the macro scores ##
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=unique_labels(y_test))
    precision_macro = classifi_report["macro avg"]["precision"]
    recall_macro = classifi_report["macro avg"]["recall"]
    auc_macro = roc_auc_score(y_test, model.predict_proba(X_test), average="macro", multi_class="ovr", labels=unique_labels(y_test))

    f1_1_improved_baseline = 1 if f1_1 > baseline_1_f1 else 0
    f1_2_improved_baseline = 1 if f1_2 > baseline_2_f1 else 0
    f1_3_improved_baseline = 1 if f1_3 > baseline_3_f1 else 0
    f1_4_improved_baseline = 1 if f1_4 > baseline_4_f1 else 0
    f1_5_improved_baseline = 1 if f1_5 > baseline_5_f1 else 0

    f1_1_no_change_baseline = 1 if f1_1 == baseline_1_f1 else 0
    f1_2_no_change_baseline = 1 if f1_2 == baseline_2_f1 else 0
    f1_3_no_change_baseline = 1 if f1_3 == baseline_3_f1 else 0
    f1_4_no_change_baseline = 1 if f1_4 == baseline_4_f1 else 0
    f1_5_no_change_baseline = 1 if f1_5 == baseline_5_f1 else 0

    RI_F1_1 = (f1_1 - baseline_1_f1) / baseline_1_f1
    RI_F1_2 = (f1_2 - baseline_2_f1) / baseline_2_f1
    RI_F1_3 = (f1_3 - baseline_3_f1) / baseline_3_f1
    RI_F1_4 = (f1_4 - baseline_4_f1) / baseline_4_f1
    RI_F1_5 = (f1_5 - baseline_5_f1) / baseline_5_f1

    num_of_classes_improved_to_baseline = f1_1_improved_baseline + f1_2_improved_baseline + f1_3_improved_baseline + f1_4_improved_baseline + f1_5_improved_baseline
    num_of_classes_degraded_to_baseline = 5 - num_of_classes_improved_to_baseline
    num_of_classes_no_change_to_baseline = f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline + f1_4_no_change_baseline + f1_5_no_change_baseline

    ## weighted relative improvement
    RI_F1_weighted = RI_F1_1 * class_1_weight + RI_F1_2 * class_2_weight + RI_F1_3 * class_3_weight + RI_F1_4 * class_4_weight + RI_F1_5 * class_5_weight

    ## check the RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5 and filter out the classes that improved ##
    all_classes = [RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    ## take the average of the improved classes ##
    if len(improved_classes) > 0:
        avg_RI_Improvement = sum(improved_classes) / len(improved_classes)
    else:
        avg_RI_Improvement = 0
    if len(degraded_classes) > 0:
        avg_RI_Degradation = sum(degraded_classes) / len(degraded_classes)
    else:
        avg_RI_Degradation = 0


    results_dict = {}

    ## For multi-class classification
    results_dict["tp_1"] = tp_1
    results_dict["tn_1"] = tn_1
    results_dict["fp_1"] = fp_1
    results_dict["fn_1"] = fn_1
    results_dict["support_1"] = classifi_report["1"]["support"]


    results_dict["tp_2"] = tp_2
    results_dict["tn_2"] = tn_2
    results_dict["fp_2"] = fp_2
    results_dict["fn_2"] = fn_2
    results_dict["support_2"] = classifi_report["2"]["support"]

    results_dict["tp_3"] = tp_3
    results_dict["tn_3"] = tn_3
    results_dict["fp_3"] = fp_3
    results_dict["fn_3"] = fn_3
    results_dict["support_3"] = classifi_report["3"]["support"]

    results_dict["tp_4"] = tp_4
    results_dict["tn_4"] = tn_4
    results_dict["fp_4"] = fp_4
    results_dict["fn_4"] = fn_4
    results_dict["support_4"] = classifi_report["4"]["support"]

    results_dict["tp_5"] = tp_5
    results_dict["tn_5"] = tn_5
    results_dict["fp_5"] = fp_5
    results_dict["fn_5"] = fn_5
    results_dict["support_5"] = classifi_report["5"]["support"]

    results_dict["f1_1"] = f1_1
    results_dict["f1_2"] = f1_2
    results_dict["f1_3"] = f1_3
    results_dict["f1_4"] = f1_4
    results_dict["f1_5"] = f1_5

    results_dict["precision_1"] = precision_1
    results_dict["precision_2"] = precision_2
    results_dict["precision_3"] = precision_3
    results_dict["precision_4"] = precision_4
    results_dict["precision_5"] = precision_5

    results_dict["recall_1"] = recall_1
    results_dict["recall_2"] = recall_2
    results_dict["recall_3"] = recall_3
    results_dict["recall_4"] = recall_4
    results_dict["recall_5"] = recall_5

    results_dict["auc_1"] = auc_1
    results_dict["auc_2"] = auc_2
    results_dict["auc_3"] = auc_3
    results_dict["auc_4"] = auc_4
    results_dict["auc_5"] = auc_5

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

    ## this is for internal use ##
    results_dict["y_predict_proba"] = model.predict_proba(X_test)

    ## compare with the baseline ##
    results_dict["f1_1 > baseline"] = f1_1_improved_baseline
    results_dict["f1_2 > baseline"] = f1_2_improved_baseline
    results_dict["f1_3 > baseline"] = f1_3_improved_baseline
    results_dict["f1_4 > baseline"] = f1_4_improved_baseline
    results_dict["f1_5 > baseline"] = f1_5_improved_baseline

    results_dict["f1_1 = baseline"] = f1_1_no_change_baseline
    results_dict["f1_2 = baseline"] = f1_2_no_change_baseline
    results_dict["f1_3 = baseline"] = f1_3_no_change_baseline
    results_dict["f1_4 = baseline"] = f1_4_no_change_baseline
    results_dict["f1_5 = baseline"] = f1_5_no_change_baseline

    results_dict["#_of_classes_improved_(f1>base)"] = num_of_classes_improved_to_baseline
    results_dict["#_of_classes_degraded_(f1<base)"] = num_of_classes_degraded_to_baseline
    results_dict["#_of_classes_no_change_(f1=base)"] = num_of_classes_no_change_to_baseline

    results_dict["RI_F1_1"] = RI_F1_1
    results_dict["RI_F1_2"] = RI_F1_2
    results_dict["RI_F1_3"] = RI_F1_3
    results_dict["RI_F1_4"] = RI_F1_4
    results_dict["RI_F1_5"] = RI_F1_5
    results_dict["RI_F1_weighted"] = RI_F1_weighted

    results_dict["Avg_RI_Improvement"] = avg_RI_Improvement
    results_dict["Avg_RI_Degradation"] = avg_RI_Degradation

    return results_dict




def aggregate_results(results, target):
    """
    aggregate the results from all the folds
    """
    overall_f1_1 = 0
    overall_f1_2 = 0
    overall_f1_3 = 0
    overall_f1_4 = 0
    overall_f1_5 = 0

    overall_auc_1 = 0
    overall_auc_2 = 0
    overall_auc_3 = 0
    overall_auc_4 = 0
    overall_auc_5 = 0

    tp_1_all = 0
    tn_1_all = 0
    fp_1_all = 0
    fn_1_all = 0

    tp_2_all = 0
    tn_2_all = 0
    fp_2_all = 0
    fn_2_all = 0

    tp_3_all = 0
    tn_3_all = 0
    fp_3_all = 0
    fn_3_all = 0

    tp_4_all = 0
    tn_4_all = 0
    fp_4_all = 0
    fn_4_all = 0

    tp_5_all = 0
    tn_5_all = 0
    fp_5_all = 0
    fn_5_all = 0

    y_index_all =[]
    y_pred_all = []
    y_true_all = []
    y_predict_proba_all = []

    for key, value in results.items():
        y_true_all.extend(value["y_actual"])
    

    for key, value in results.items():
        y_index_all.extend(value["y_test_index"])
        y_pred_all.extend(value["y_pred"])

        ## this is for internal use
        y_predict_proba_all.extend(value["y_predict_proba"])
        
        tp_1_all += value["tp_1"]
        tn_1_all += value["tn_1"]
        fp_1_all += value["fp_1"]
        fn_1_all += value["fn_1"]

        tp_2_all += value["tp_2"]
        tn_2_all += value["tn_2"]
        fp_2_all += value["fp_2"]
        fn_2_all += value["fn_2"]

        tp_3_all += value["tp_3"]
        tn_3_all += value["tn_3"]
        fp_3_all += value["fp_3"]
        fn_3_all += value["fn_3"]

        tp_4_all += value["tp_4"]
        tn_4_all += value["tn_4"]
        fp_4_all += value["fp_4"]
        fn_4_all += value["fn_4"]

        tp_5_all += value["tp_5"]
        tn_5_all += value["tn_5"]
        fp_5_all += value["fp_5"]
        fn_5_all += value["fn_5"]

    overall_f1_1 = f1_score(y_true_all, y_pred_all, average=None)[0]
    overall_f1_2 = f1_score(y_true_all, y_pred_all, average=None)[1]
    overall_f1_3 = f1_score(y_true_all, y_pred_all, average=None)[2]
    overall_f1_4 = f1_score(y_true_all, y_pred_all, average=None)[3]
    overall_f1_5 = f1_score(y_true_all, y_pred_all, average=None)[4]

    overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[0]
    overall_auc_2 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[1]
    overall_auc_3 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[2]
    overall_auc_4 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[3]
    overall_auc_5 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[4]

    overall_precision_1 = precision_score(y_true_all, y_pred_all, average=None)[0]
    overall_precision_2 = precision_score(y_true_all, y_pred_all, average=None)[1]
    overall_precision_3 = precision_score(y_true_all, y_pred_all, average=None)[2]
    overall_precision_4 = precision_score(y_true_all, y_pred_all, average=None)[3]
    overall_precision_5 = precision_score(y_true_all, y_pred_all, average=None)[4]

    overall_recall_1 = recall_score(y_true_all, y_pred_all, average=None)[0]
    overall_recall_2 = recall_score(y_true_all, y_pred_all, average=None)[1]
    overall_recall_3 = recall_score(y_true_all, y_pred_all, average=None)[2]
    overall_recall_4 = recall_score(y_true_all, y_pred_all, average=None)[3]
    overall_recall_5 = recall_score(y_true_all, y_pred_all, average=None)[4]
    
    overall_f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted")
    overall_auc_weighted = roc_auc_score(y_true_all, y_predict_proba_all, average="weighted", multi_class="ovr", labels=[1,2,3,4,5])
    overall_precision_weighted = precision_score(y_true_all, y_pred_all, average="weighted")
    overall_recall_weighted = recall_score(y_true_all, y_pred_all, average="weighted")

    overall_f1_macro = f1_score(y_true_all, y_pred_all, average="macro")
    overall_auc_macro = roc_auc_score(y_true_all, y_predict_proba_all, average="macro", multi_class="ovr", labels=[1,2,3,4,5])
    overall_precision_macro = precision_score(y_true_all, y_pred_all, average="macro")
    overall_recall_macro = recall_score(y_true_all, y_pred_all, average="macro")

    ## get baseline f1 scores and class weights ##
    baseline_f1_1, baseline_f1_2, baseline_f1_3, baseline_f1_4, baseline_f1_5, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight = getBestBaselineModel(target)

    ## compare the improvement to the baseline ##
    f1_1_improved_baseline = 1 if overall_f1_1 > baseline_f1_1 else 0
    f1_2_improved_baseline = 1 if overall_f1_2 > baseline_f1_2 else 0
    f1_3_improved_baseline = 1 if overall_f1_3 > baseline_f1_3 else 0
    f1_4_improved_baseline = 1 if overall_f1_4 > baseline_f1_4 else 0
    f1_5_improved_baseline = 1 if overall_f1_5 > baseline_f1_5 else 0

    f1_1_no_change_baseline = 1 if overall_f1_1 == baseline_f1_1 else 0
    f1_2_no_change_baseline = 1 if overall_f1_2 == baseline_f1_2 else 0
    f1_3_no_change_baseline = 1 if overall_f1_3 == baseline_f1_3 else 0
    f1_4_no_change_baseline = 1 if overall_f1_4 == baseline_f1_4 else 0
    f1_5_no_change_baseline = 1 if overall_f1_5 == baseline_f1_5 else 0

    RI_F1_1 = (overall_f1_1 - baseline_f1_1) / baseline_f1_1
    RI_F1_2 = (overall_f1_2 - baseline_f1_2) / baseline_f1_2
    RI_F1_3 = (overall_f1_3 - baseline_f1_3) / baseline_f1_3
    RI_F1_4 = (overall_f1_4 - baseline_f1_4) / baseline_f1_4
    RI_F1_5 = (overall_f1_5 - baseline_f1_5) / baseline_f1_5

    num_of_classes_improved_to_baseline = f1_1_improved_baseline + f1_2_improved_baseline + f1_3_improved_baseline + f1_4_improved_baseline + f1_5_improved_baseline
    num_of_classes_degraded_to_baseline = 5 - num_of_classes_improved_to_baseline
    num_of_classes_no_change_to_baseline = f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline + f1_4_no_change_baseline + f1_5_no_change_baseline

    ## weighted relative improvement
    RI_F1_weighted = RI_F1_1 * class_1_weight + RI_F1_2 * class_2_weight + RI_F1_3 * class_3_weight + RI_F1_4 * class_4_weight + RI_F1_5 * class_5_weight

    ## check the RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5 and filter out the classes that improved ##
    all_classes = [RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    ## take the average of the improved classes ##
    if len(improved_classes) > 0:
        avg_RI_Improvement = sum(improved_classes) / len(improved_classes)
    else:
        avg_RI_Improvement = 0
    if len(degraded_classes) > 0:
        avg_RI_Degradation = sum(degraded_classes) / len(degraded_classes)
    else:
        avg_RI_Degradation = 0    

    
    num_instances = len(y_true_all)

    ## remove np.int64 from the array
    y_index_all = [int(num) for num in y_index_all]
    y_true_all = [int(num) for num in y_true_all]
    y_pred_all = [int(num) for num in y_pred_all]

    return overall_f1_weighted, overall_f1_macro, overall_precision_weighted, overall_precision_macro, overall_recall_weighted, overall_recall_macro, overall_f1_1, overall_f1_2, overall_f1_3, overall_f1_4, overall_f1_5, overall_precision_1, overall_precision_2, overall_precision_3, overall_precision_4, overall_precision_5, overall_recall_1, overall_recall_2, overall_recall_3, overall_recall_4, overall_recall_5, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, tp_4_all, tn_4_all, fp_4_all, fn_4_all, tp_5_all, tn_5_all, fp_5_all, fn_5_all, num_instances, y_index_all, y_true_all, y_pred_all, num_of_classes_improved_to_baseline, num_of_classes_degraded_to_baseline, num_of_classes_no_change_to_baseline, RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, avg_RI_Improvement, avg_RI_Degradation, RI_F1_weighted, f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, overall_auc_1, overall_auc_2, overall_auc_3, overall_auc_4, overall_auc_5, overall_auc_weighted, overall_auc_macro



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
            "random_state": [configs_DS3.RANDOM_SEED]
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
            "n_neighbors": [1, 3, 5, 7],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"],
            "algorithm": ["ball_tree", "kd_tree"],
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
            "max_features": [None],
            "min_impurity_decrease": [0.001, 0.01 ],
            "max_depth": [None, 10],
            # "max_depth": [30, 50],
            # "bootstrap": [True],
            # "min_samples_split": [10, 15],
            # "min_samples_leaf": [4, 5],
            "random_state": [configs_DS3.RANDOM_SEED]
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
            "random_state": [configs_DS3.RANDOM_SEED]
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
            "random_state": [configs_DS3.RANDOM_SEED],
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
            # "var_smoothing": [],
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = GaussianNB()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = GaussianNB(**parameters)
    
    return model, param_grid

def dict_data_generator(model, iteration, hyperparameters, target, drop_duplicates, use_oversampling, dataset, K, fs_method, y_test_index, y_actual,y_pred, tp_1, tn_1, fp_1, fn_1, tp_2, tn_2, fp_2, fn_2, tp_3, tn_3, fp_3, fn_3, tp_4, tn_4, fp_4, fn_4, tp_5, tn_5, fp_5, fn_5,    f1_1, f1_2, f1_3, f1_4, f1_5, support_1, support_2, support_3, support_4, support_5, precision_1, precision_2, precision_3, precision_4, precision_5, recall_1, recall_2, recall_3, recall_4, recall_5, n_instances, precision_weighted, recall_weighted, f1_weighted, f1_macro,  experiment, precision_macro, recall_macro, f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, num_of_classes_improved_to_baseline, num_of_classes_detegraded_to_baseline, num_of_classes_no_change_to_baseline ,RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, RI_F1_weighted, Avg_RI_Improvement, Avg_RI_Degradation, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, auc_1, auc_2, auc_3, auc_4, auc_5, auc_weighted, auc_macro):
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

    csv_data_dict["tp_4"] = tp_4
    csv_data_dict["tn_4"] = tn_4
    csv_data_dict["fp_4"] = fp_4
    csv_data_dict["fn_4"] = fn_4
    csv_data_dict["support_4"] = support_4

    csv_data_dict["tp_5"] = tp_5
    csv_data_dict["tn_5"] = tn_5
    csv_data_dict["fp_5"] = fp_5
    csv_data_dict["fn_5"] = fn_5    
    csv_data_dict["support_5"] = support_5 

    csv_data_dict["f1_1"] = f1_1
    csv_data_dict["f1_2"] = f1_2
    csv_data_dict["f1_3"] = f1_3
    csv_data_dict["f1_4"] = f1_4
    csv_data_dict["f1_5"] = f1_5

    csv_data_dict["auc_1"] = auc_1
    csv_data_dict["auc_2"] = auc_2
    csv_data_dict["auc_3"] = auc_3
    csv_data_dict["auc_4"] = auc_4
    csv_data_dict["auc_5"] = auc_5

    csv_data_dict["precision_1"] = precision_1
    csv_data_dict["precision_2"] = precision_2
    csv_data_dict["precision_3"] = precision_3
    csv_data_dict["precision_4"] = precision_4
    csv_data_dict["precision_5"] = precision_5

    csv_data_dict["recall_1"] = recall_1
    csv_data_dict["recall_2"] = recall_2
    csv_data_dict["recall_3"] = recall_3
    csv_data_dict["recall_4"] = recall_4
    csv_data_dict["recall_5"] = recall_5

    csv_data_dict["n_instances"] = n_instances

    csv_data_dict["f1_weighted"] = f1_weighted
    csv_data_dict["precision_weighted"] = precision_weighted
    csv_data_dict["auc_weighted"] = auc_weighted
    csv_data_dict["recall_weighted"] = recall_weighted

    csv_data_dict["f1_macro"] = f1_macro
    csv_data_dict["auc_macro"] = auc_macro
    csv_data_dict["precision_macro"] = precision_macro
    csv_data_dict["recall_macro"] = recall_macro

    csv_data_dict["f1_1 > baseline"] = f1_1_improved_baseline
    csv_data_dict["f1_2 > baseline"] = f1_2_improved_baseline
    csv_data_dict["f1_3 > baseline"] = f1_3_improved_baseline
    csv_data_dict["f1_4 > baseline"] = f1_4_improved_baseline
    csv_data_dict["f1_5 > baseline"] = f1_5_improved_baseline

    csv_data_dict["f1_1 = baseline"] = f1_1_no_change_baseline
    csv_data_dict["f1_2 = baseline"] = f1_2_no_change_baseline
    csv_data_dict["f1_3 = baseline"] = f1_3_no_change_baseline
    csv_data_dict["f1_4 = baseline"] = f1_4_no_change_baseline
    csv_data_dict["f1_5 = baseline"] = f1_5_no_change_baseline

    csv_data_dict["#_of_classes_improved_(f1>base)"] = num_of_classes_improved_to_baseline
    csv_data_dict["#_of_classes_degraded_(f1<base)"] = num_of_classes_detegraded_to_baseline
    csv_data_dict["#_of_classes_no_change_(f1=base)"] = num_of_classes_no_change_to_baseline

    csv_data_dict["RI_F1_1"] = RI_F1_1
    csv_data_dict["RI_F1_2"] = RI_F1_2
    csv_data_dict["RI_F1_3"] = RI_F1_3
    csv_data_dict["RI_F1_4"] = RI_F1_4
    csv_data_dict["RI_F1_5"] = RI_F1_5
    csv_data_dict["RI_F1_weighted"] = RI_F1_weighted

    csv_data_dict["Avg_RI_Improvement"] = Avg_RI_Improvement
    csv_data_dict["Avg_RI_Degradation"] = Avg_RI_Degradation

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
    complete_df = pd.read_csv(configs_DS3.ROOT_PATH + "/" + configs_DS3.DATA_PATH)
    output_file = "DS3_TASK1.csv"
    
    ## write header
    with open(configs_DS3.ROOT_PATH + "/results/" + output_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(configs_DS3.ROOT_PATH + "/featureselection/experiments_DS3_filtered.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=configs_DS3.RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=configs_DS3.RANDOM_SEED)
    # outer_cv = LeaveOneOut()

    models = [ "bayes_network","knn_classifier", "logisticregression",  "mlp_classifier", "svc", "randomForest_classifier" ]

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
            ## convert target values to integers
            target_y = target_y.astype(int)
        

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
                ## Why not StandardScaler? - because it is sensitive to outliers. Also we have one feature CR which all the values are constant. 
                pipeline = Pipeline(steps = [('scaler', RobustScaler()), (model_name, model)])
                
                ## CONFIG 1 ## - apply over sampling
                if use_oversampling: # if use RandomOverSampling
                        # ros = RandomOverSampler(random_state=configs_DS3.RANDOM_SEED)
                        smo = SMOTE(random_state=configs_DS3.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', RobustScaler()),    
                        # ('ros', ros),
                        ('smo', smo),
                        (model_name, model)])

                ## CONFIG 2 ## - remove duplicates from training set
                config = {"drop_duplicates": drop_duplicates}
                
                best_params, best_score_ = custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, inner_cv, config, target)
                LOGGER.info("Best param searching for fold {} for code features...".format(fold))
                # gridSearchCV = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
                # gridSearchCV.fit(X_train, y_train.values.ravel())
                # best_params = gridSearchCV.best_params_
                
                # best_params = {key.replace(model_name + "__", ""):value for key, value in best_params.items()} 
                
                ## since we are using a set, we need to convert the dict to a hashable type
                best_hyperparams.add((frozenset(best_params)))



            #############################################
            # Train and Test with best hyperparameters ##
            #############################################
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
                    
                    if experiment['use_oversampling']:
                        # ros = RandomOverSampler(random_state=configs_DS3.RANDOM_SEED)
                        smo = SMOTE(random_state=configs_DS3.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        ('scaler', RobustScaler()),    
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
                        tp_4=fold_results_["tp_4"],
                        tn_4=fold_results_["tn_4"],
                        fp_4=fold_results_["fp_4"],
                        fn_4=fold_results_["fn_4"],
                        support_4=fold_results_["support_4"],
                        tp_5=fold_results_["tp_5"],
                        tn_5=fold_results_["tn_5"],
                        fp_5=fold_results_["fp_5"],
                        fn_5=fold_results_["fn_5"],
                        support_5=fold_results_["support_5"],
                        n_instances=fold_results_["n_instances"],
                        f1_1=fold_results_["f1_1"],
                        f1_2=fold_results_["f1_2"],
                        f1_3=fold_results_["f1_3"],
                        f1_4=fold_results_["f1_4"],
                        f1_5=fold_results_["f1_5"],
                        auc_1=fold_results_["auc_1"],
                        auc_2=fold_results_["auc_2"],
                        auc_3=fold_results_["auc_3"],
                        auc_4=fold_results_["auc_4"],
                        auc_5=fold_results_["auc_5"],
                        precision_1=fold_results_["precision_1"],
                        precision_2=fold_results_["precision_2"],
                        precision_3=fold_results_["precision_3"],
                        precision_4=fold_results_["precision_4"],
                        precision_5=fold_results_["precision_5"],
                        recall_1=fold_results_["recall_1"],
                        recall_2=fold_results_["recall_2"],
                        recall_3=fold_results_["recall_3"],
                        recall_4=fold_results_["recall_4"],
                        recall_5=fold_results_["recall_5"],
                        
                        f1_weighted=fold_results_["f1_weighted"],
                        precision_weighted=fold_results_["precision_weighted"],
                        auc_weighted=fold_results_["auc_weighted"],
                        recall_weighted=fold_results_["recall_weighted"],
                        f1_macro=fold_results_["f1_macro"],
                        auc_macro=fold_results_["auc_macro"],
                        precision_macro=fold_results_["precision_macro"],
                        recall_macro=fold_results_["recall_macro"],
                        f1_1_improved_baseline=fold_results_["f1_1 > baseline"],
                        f1_2_improved_baseline=fold_results_["f1_2 > baseline"],
                        f1_3_improved_baseline=fold_results_["f1_3 > baseline"],
                        f1_4_improved_baseline=fold_results_["f1_4 > baseline"],
                        f1_5_improved_baseline=fold_results_["f1_5 > baseline"],
                        f1_1_no_change_baseline=fold_results_["f1_1 = baseline"],
                        f1_2_no_change_baseline=fold_results_["f1_2 = baseline"],
                        f1_3_no_change_baseline=fold_results_["f1_3 = baseline"],
                        f1_4_no_change_baseline=fold_results_["f1_4 = baseline"],
                        f1_5_no_change_baseline=fold_results_["f1_5 = baseline"],
                        num_of_classes_improved_to_baseline=fold_results_["#_of_classes_improved_(f1>base)"],
                        num_of_classes_detegraded_to_baseline=fold_results_["#_of_classes_degraded_(f1<base)"],
                        num_of_classes_no_change_to_baseline=fold_results_["#_of_classes_no_change_(f1=base)"],
                        RI_F1_1=fold_results_["RI_F1_1"],
                        RI_F1_2=fold_results_["RI_F1_2"],
                        RI_F1_3=fold_results_["RI_F1_3"],
                        RI_F1_4=fold_results_["RI_F1_4"],
                        RI_F1_5=fold_results_["RI_F1_5"],
                        RI_F1_weighted=fold_results_["RI_F1_weighted"],
                        Avg_RI_Improvement=fold_results_["Avg_RI_Improvement"],
                        Avg_RI_Degradation=fold_results_["Avg_RI_Degradation"],
                        experiment=experiment["exp_id"])

                    dict_to_csv(configs_DS3.ROOT_PATH + "/results/" + output_file, csv_data_dict1)



            
                ## aggregate the results from all the folds
                overall_f1_weighted, overall_f1_macro, overall_precision_weighted, overall_precision_macro, overall_recall_weighted, overall_recall_macro, overall_f1_1, overall_f1_2, overall_f1_3, overall_f1_4, overall_f1_5, overall_precision_1, overall_precision_2, overall_precision_3, overall_precision_4, overall_precision_5, overall_recall_1, overall_recall_2, overall_recall_3, overall_recall_4, overall_recall_5, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, tp_4_all, tn_4_all, fp_4_all, fn_4_all, tp_5_all, tn_5_all, fp_5_all, fn_5_all, num_instances, y_index_all, y_true_all, y_pred_all, num_of_classes_improved_to_baseline, num_of_classes_degraded_to_baseline, num_of_classes_no_change_to_baseline, RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, Avg_RI_Improvement, Avg_RI_Degradation, RI_F1_weighted, f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, overall_auc_1, overall_auc_2, overall_auc_3, overall_auc_4, overall_auc_5, overall_auc_weighted, overall_auc_macro = aggregate_results(all_fold_results, target) 
                
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

                    tp_4=tp_4_all,
                    tn_4=tn_4_all,
                    fp_4=fp_4_all,
                    fn_4=fn_4_all,
                    support_4=tp_4_all + fn_4_all,

                    tp_5=tp_5_all,
                    tn_5=tn_5_all,
                    fp_5=fp_5_all,
                    fn_5=fn_5_all,
                    support_5=tp_5_all + fn_5_all,

                    f1_1=overall_f1_1,
                    f1_2=overall_f1_2,
                    f1_3=overall_f1_3,
                    f1_4=overall_f1_4,
                    f1_5=overall_f1_5,

                    auc_1=overall_auc_1,
                    auc_2=overall_auc_2,
                    auc_3=overall_auc_3,
                    auc_4=overall_auc_4,
                    auc_5=overall_auc_5,

                    precision_1=overall_precision_1,
                    precision_2=overall_precision_2,
                    precision_3=overall_precision_3,
                    precision_4=overall_precision_4,
                    precision_5=overall_precision_5,

                    recall_1=overall_recall_1,
                    recall_2=overall_recall_2,
                    recall_3=overall_recall_3,
                    recall_4=overall_recall_4,
                    recall_5=overall_recall_5,

                    n_instances=num_instances,

                    f1_weighted=overall_f1_weighted,
                    precision_weighted=overall_precision_weighted,
                    auc_weighted=overall_auc_weighted,
                    recall_weighted=overall_recall_weighted,

                    f1_macro=overall_f1_macro,
                    auc_macro=overall_auc_macro,
                    precision_macro=overall_precision_macro,
                    recall_macro=overall_recall_macro,

                    f1_1_improved_baseline=f1_1_improved_baseline,
                    f1_2_improved_baseline=f1_2_improved_baseline,
                    f1_3_improved_baseline=f1_3_improved_baseline,
                    f1_4_improved_baseline=f1_4_improved_baseline,
                    f1_5_improved_baseline=f1_5_improved_baseline,

                    f1_1_no_change_baseline=f1_1_no_change_baseline,
                    f1_2_no_change_baseline=f1_2_no_change_baseline,
                    f1_3_no_change_baseline=f1_3_no_change_baseline,
                    f1_4_no_change_baseline=f1_4_no_change_baseline,
                    f1_5_no_change_baseline=f1_5_no_change_baseline,

                    num_of_classes_improved_to_baseline=num_of_classes_improved_to_baseline,
                    num_of_classes_detegraded_to_baseline=num_of_classes_degraded_to_baseline,
                    num_of_classes_no_change_to_baseline=num_of_classes_no_change_to_baseline,

                    RI_F1_1=RI_F1_1,
                    RI_F1_2=RI_F1_2,
                    RI_F1_3=RI_F1_3,
                    RI_F1_4=RI_F1_4,
                    RI_F1_5=RI_F1_5,

                    RI_F1_weighted=RI_F1_weighted,

                    Avg_RI_Improvement=Avg_RI_Improvement,
                    Avg_RI_Degradation=Avg_RI_Degradation,


                    experiment=experiment["exp_id"])
   
                dict_to_csv(configs_DS3.ROOT_PATH + "/results/" + output_file, csv_data_dict2)

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

if __name__ == "__main__":
    main()