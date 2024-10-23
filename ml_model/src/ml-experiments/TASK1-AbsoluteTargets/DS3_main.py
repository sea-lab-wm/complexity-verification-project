import sys
import logging
from utils import configs_TASK1_DS3 as configs
sys.path.append(configs.ROOT_PATH)


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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import ParameterGrid

from sklearn.utils.parallel import Parallel, delayed
from joblib import parallel_backend ## train the models in parallel

from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import os
from mlxtend.plotting import plot_decision_regions

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

    "precision_weighted": "",
    "recall_weighted": "",
    "f1_weighted": "",
    "auc_weighted": "",

    "precision_macro":"",
    "recall_macro": "",
    "f1_macro": "",
    "auc_macro": "",

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

    "F1_weighted > baseline": 0,
    "RI=(F1_weighted-baseline)/baseline": 0,

    "RI_F1_1": "",
    "RI_F1_2": "",
    "RI_F1_3": "",
    "RI_F1_4": "",
    "RI_F1_5": "",
    "RI_weighted": "",
    
    "majority_classes_Improved": "", ## Majority classes(2,3,4) improved
    "minority_classes_Improved": "", ## Minority classes(1,5) improved

    "avg_RI_Improved_Majority_Classes":	"", ## All Majority classes(2,3,4) improved 
    "avg_RI_Degraded_Majority_Classes": "",	
    "avg_RI_Majority_Classes": "",
    "avg_RI_Improved_Minority_Classes": "",	
    "avg_RI_Degraded_Minority_Classes":"",
    "avg_RI_Minority_Classes":"",

    "avg_RI_Improvement": "", # average relative improvement for all classes that improved
    "avg_RI_Degradation": "", # average relative improvement for all classes that degraded
    
    "experiment": ""
}

def custom_grid_search_cv(model_name, pipeline, param_grid, X_train, y_train, cv, config, target):
    ## all the permutations of the hyperparameters ##
    candidate_params = list(ParameterGrid(param_grid))

    drop_duplicates = config["drop_duplicates"]
    best_hyperparameters_ = {} ## keep best hyperparameters for each fold and corresponding f1_weighted score

    # Process each fold in parallel
    results=Parallel(n_jobs=30)(delayed(process_fold)(fold, train_index, test_index, candidate_params, drop_duplicates, model_name, pipeline, X_train, y_train, target)
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

        ### SPECIAL CODE ##
        # X_test = model.named_steps.scaler.transform(X_val_fold.values)
        # results = {}
        # y_pred = model.predict(X_test)
        # f1_weighted = f1_score(y_val_fold, y_pred, average="weighted", labels=unique_labels(y_val_fold), zero_division=np.nan)
        # results["f1_weighted"] = f1_weighted
        ### SPECIAL CODE ##

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
    """
    train the model parallelly using Joblib
    """
    # # Use a Joblib context manager to train the classifier in parallel
    # with parallel_backend('loky', n_jobs=-1):
    model.fit(X_train.values, y_train.values.ravel())
    return model

def getBestBaselineModel(target):
    if target == "readability_level":
        baseline_1_f1 = 0.0734711
        baseline_2_f1 = 0.2050413
        baseline_3_f1 = 0.2677686
        baseline_4_f1 = 0.2719008
        baseline_5_f1 = 0.1818182

        baseline_f1_weighted = 0.2261279

        class_1_weight = 889/12100
        class_2_weight = 2481/12100
        class_3_weight = 3240/12100
        class_4_weight = 3290/12100
        class_5_weight = 2200/12100
        

    return baseline_1_f1, baseline_2_f1, baseline_3_f1, baseline_4_f1, baseline_5_f1, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight, baseline_f1_weighted       


def evaluate(model, X_test, y_test, target):
    ## Transform the test data to Scaler ##
    # X_test = model.named_steps.scaler.transform(X_test.values)
    X_test = X_test.values
    
    ## predict on the test split ##
    y_pred = model.predict(X_test)

    classifi_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) ## this ensures that the f1 score is 0 if there are no predicted instances
    prediction_report = multilabel_confusion_matrix (y_test, y_pred, labels=unique_labels(y_test))

    ## get baseline f1 scores and class weights ##
    baseline_1_f1, baseline_2_f1, baseline_3_f1, baseline_4_f1, baseline_5_f1, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight, baseline_f1_weighted = getBestBaselineModel(target)

    ## For multi-class classification
    f1_1 = classifi_report["1"]["f1-score"]
    f1_2 = classifi_report["2"]["f1-score"]
    f1_3 = classifi_report["3"]["f1-score"]
    f1_4 = classifi_report["4"]["f1-score"]
    f1_5 = classifi_report["5"]["f1-score"]

    # auc_1 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[0]
    # auc_2 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[1]
    # auc_3 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[2]
    # auc_4 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[3]
    # auc_5 = roc_auc_score(y_test, model.predict_proba(X_test), average=None, multi_class="ovr", labels=unique_labels(y_test))[4]

    ## TODO: Need to remove AUC from result calculation ##
    auc_1 = 0
    auc_2 = 0
    auc_3 = 0
    auc_4 = 0
    auc_5 = 0

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
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=unique_labels(y_test), zero_division=0) ## zero_division=0 to avoid nan values to include in the average
    precision_weighted = classifi_report["weighted avg"]["precision"]
    recall_weighted = classifi_report["weighted avg"]["recall"]
    # auc_weighted = roc_auc_score(y_test, model.predict_proba(X_test), average="weighted", multi_class="ovr", labels=unique_labels(y_test))
    auc_weighted = 0
    
    ## compute the macro scores ##
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=unique_labels(y_test), zero_division=0)
    precision_macro = classifi_report["macro avg"]["precision"]
    recall_macro = classifi_report["macro avg"]["recall"]
    # auc_macro = roc_auc_score(y_test, model.predict_proba(X_test), average="macro", multi_class="ovr", labels=unique_labels(y_test))
    auc_macro = 0

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

    F1_weighted_improved_over_baseline = 1 if f1_weighted > baseline_f1_weighted else 0
    RI_F1_weighted = (f1_weighted - baseline_f1_weighted) / baseline_f1_weighted

    ## relative improvement for each class ##
    RI_F1_1 = (f1_1 - baseline_1_f1) / baseline_1_f1 if f1_1 != 0 else "" ## keep the value empty if the f1 score is 0
    RI_F1_2 = (f1_2 - baseline_2_f1) / baseline_2_f1 if f1_2 != 0 else ""
    RI_F1_3 = (f1_3 - baseline_3_f1) / baseline_3_f1 if f1_3 != 0 else ""
    RI_F1_4 = (f1_4 - baseline_4_f1) / baseline_4_f1 if f1_4 != 0 else ""
    RI_F1_5 = (f1_5 - baseline_5_f1) / baseline_5_f1 if f1_5 != 0 else ""

    
    if (f1_2 > baseline_2_f1 and f1_3 > baseline_3_f1 and f1_4 > baseline_4_f1):
        Majority_classes_Improved = "ALL MAJORITY"
    elif (f1_2 > baseline_2_f1 or f1_3 > baseline_3_f1 or f1_4 > baseline_4_f1):
        Majority_classes_Improved = "SOME MAJORITY"
    else:
        Majority_classes_Improved = "NO MAJORITY"

    if (f1_1 > baseline_1_f1 and f1_5 > baseline_5_f1):
        Minority_classes_Improved = "ALL MINORITY"
    elif (f1_1 > baseline_1_f1 or f1_5 > baseline_5_f1):
        Minority_classes_Improved = "SOME MINORITY"
    else:
        Minority_classes_Improved = "NO MINORITY"    

    num_of_classes_improved_to_baseline = f1_1_improved_baseline + f1_2_improved_baseline + f1_3_improved_baseline + f1_4_improved_baseline + f1_5_improved_baseline
    num_of_classes_degraded_to_baseline = 5 - num_of_classes_improved_to_baseline
    num_of_classes_no_change_to_baseline = f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline + f1_4_no_change_baseline + f1_5_no_change_baseline

    ## weighted relative improvement
    FIXED_RI_F1_1 = 0 if RI_F1_1 == "" else RI_F1_1
    FIXED_RI_F1_2 = 0 if RI_F1_2 == "" else RI_F1_2
    FIXED_RI_F1_3 = 0 if RI_F1_3 == "" else RI_F1_3
    FIXED_RI_F1_4 = 0 if RI_F1_4 == "" else RI_F1_4
    FIXED_RI_F1_5 = 0 if RI_F1_5 == "" else RI_F1_5
    
    RI_weighted = FIXED_RI_F1_1 * class_1_weight + FIXED_RI_F1_2 * class_2_weight + FIXED_RI_F1_3 * class_3_weight + FIXED_RI_F1_4 * class_4_weight + FIXED_RI_F1_5 * class_5_weight

    ## check the RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5 and filter out the classes that improved ##
    all_classes = [FIXED_RI_F1_1, FIXED_RI_F1_2, FIXED_RI_F1_3, FIXED_RI_F1_4, FIXED_RI_F1_5]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    all_majority_classes = [FIXED_RI_F1_2, FIXED_RI_F1_3, FIXED_RI_F1_4]
    improved_majority_classes = [x for x in all_majority_classes if x > 0]
    degraded_majority_classes = [x for x in all_majority_classes if x < 0]

    all_minority_classes = [FIXED_RI_F1_1, FIXED_RI_F1_5]
    improved_minority_classes = [x for x in all_minority_classes if x > 0]
    degraded_minority_classes = [x for x in all_minority_classes if x < 0]

    ## take the average of the improved classes ##
    if len(improved_classes) > 0:
        avg_RI_Improvement = sum(improved_classes) / len(improved_classes)
    else:
        avg_RI_Improvement = ""
    if len(degraded_classes) > 0:
        avg_RI_Degradation = sum(degraded_classes) / len(degraded_classes)
    else:
        avg_RI_Degradation = ""


    if len(improved_majority_classes) > 0:
        avg_RI_Improved_Majority_Classes = sum(improved_majority_classes) / len(improved_majority_classes)
    else:
        avg_RI_Improved_Majority_Classes = ""
    if len(degraded_majority_classes) > 0:
        avg_RI_Degraded_Majority_Classes = sum(degraded_majority_classes) / len(degraded_majority_classes)
    else:
        avg_RI_Degraded_Majority_Classes = ""

    if len(improved_minority_classes) > 0:
        avg_RI_Improved_Minority_Classes = sum(improved_minority_classes) / len(improved_minority_classes)
    else:
        avg_RI_Improved_Minority_Classes = ""
    if len(degraded_minority_classes) > 0:
        avg_RI_Degraded_Minority_Classes = sum(degraded_minority_classes) / len(degraded_minority_classes)
    else:
        avg_RI_Degraded_Minority_Classes = ""


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
    ## TODO: Need to remove this ##
    # results_dict["y_predict_proba"] = model.predict_proba(X_test)
    results_dict["y_predict_proba"] = np.zeros((len(y_test)))

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

    results_dict["F1_weighted > baseline"] = F1_weighted_improved_over_baseline
    results_dict["RI=(F1_weighted-baseline)/baseline"] = RI_F1_weighted

    results_dict["RI_F1_1"] = RI_F1_1
    results_dict["RI_F1_2"] = RI_F1_2
    results_dict["RI_F1_3"] = RI_F1_3
    results_dict["RI_F1_4"] = RI_F1_4
    results_dict["RI_F1_5"] = RI_F1_5
    results_dict["RI_weighted"] = RI_weighted

    results_dict["majority_classes_Improved"] = Majority_classes_Improved ## Majority classes(2,3,4) improved
    results_dict["minority_classes_Improved"] = Minority_classes_Improved ## Minority classes(1,5) improved

    results_dict["avg_RI_Improved_Majority_Classes"] = avg_RI_Improved_Majority_Classes	## All Majority classes(2,3,4) improved
    results_dict["avg_RI_Degraded_Majority_Classes"] = avg_RI_Degraded_Majority_Classes	
    results_dict["avg_RI_Majority_Classes"] = np.nanmean(all_majority_classes)
    results_dict["avg_RI_Improved_Minority_Classes"]= avg_RI_Improved_Minority_Classes	
    results_dict["avg_RI_Degraded_Minority_Classes"] = avg_RI_Degraded_Minority_Classes
    results_dict["avg_RI_Minority_Classes"] = np.nanmean(all_minority_classes)

    results_dict["avg_RI_Improvement"] = avg_RI_Improvement
    results_dict["avg_RI_Degradation"] = avg_RI_Degradation


    ## validation_curve ##
    # from sklearn.model_selection import validation_curve
    # param_range = np.logspace(-7, 3, 10)
    # train_scores, test_scores = validation_curve(model, X_test, y_test, param_name="logisticregression__C", cv=5, param_range=param_range, n_jobs=-1)
    # ## plot the validation curve ##
    # plt.figure()
    # plt.semilogx(param_range, np.mean(train_scores, axis=1), label="Training score", color="darkorange")
    # plt.semilogx(param_range, np.mean(test_scores, axis=1), label="Cross-validation score", color="navy")
    # plt.title("Validation Curve with SVM")
    # plt.xlabel("C")
    # plt.ylabel("Score")
    # plt.legend(loc="best")
    # ## save in the results dir ##
    # if not os.path.exists(configs.ROOT_PATH + "/results/residuals/" + target + "/" + "lr" + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/residuals/" + target + "/" + "lr" + "/")
    # plt.savefig(configs.ROOT_PATH + "/results/residuals/" + target + "/" + "lr" + "/" + str(10) + "_" + target  + "_.png")
    



    return results_dict



def aggregate_results(results, target, model_name, K, hyperparam, model):
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

    overall_f1_1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_f1_2 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]
    overall_f1_3 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
    overall_f1_4 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]
    overall_f1_5 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)[4]

    # overall_auc_1 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[0]
    # overall_auc_2 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[1]
    # overall_auc_3 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[2]
    # overall_auc_4 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[3]
    # overall_auc_5 = roc_auc_score(y_true_all, y_predict_proba_all, average=None, multi_class="ovr", labels=[1,2,3,4,5])[4]

    ## TODO: Need to remove AUC from result calculation ##
    overall_auc_1 = 0
    overall_auc_2 = 0
    overall_auc_3 = 0
    overall_auc_4 = 0
    overall_auc_5 = 0

    overall_precision_1 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_precision_2 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]
    overall_precision_3 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
    overall_precision_4 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]
    overall_precision_5 = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)[4]

    overall_recall_1 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[0]
    overall_recall_2 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[1]
    overall_recall_3 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[2]
    overall_recall_4 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[3]
    overall_recall_5 = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)[4]
    
    overall_f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    # overall_auc_weighted = roc_auc_score(y_true_all, y_predict_proba_all, average="weighted", multi_class="ovr", labels=[1,2,3,4,5])
    overall_auc_weighted = 0
    overall_precision_weighted = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    overall_recall_weighted = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

    overall_f1_macro = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    # overall_auc_macro = roc_auc_score(y_true_all, y_predict_proba_all, average="macro", multi_class="ovr", labels=[1,2,3,4,5])
    overall_auc_macro = 0
    overall_precision_macro = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    overall_recall_macro = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    ## get baseline f1 scores and class weights ##
    baseline_f1_1, baseline_f1_2, baseline_f1_3, baseline_f1_4, baseline_f1_5, class_1_weight, class_2_weight, class_3_weight, class_4_weight, class_5_weight, baseline_f1_weighted = getBestBaselineModel(target)

    F1_weighted_improved_over_baseline = 1 if overall_f1_weighted > baseline_f1_weighted else 0
    RI_F1_weighted = (overall_f1_weighted - baseline_f1_weighted) / baseline_f1_weighted

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

    RI_F1_1 = (overall_f1_1 - baseline_f1_1) / baseline_f1_1 if overall_f1_1 != 0 else "" ## keep the value empty if the f
    RI_F1_2 = (overall_f1_2 - baseline_f1_2) / baseline_f1_2 if overall_f1_2 != 0 else ""
    RI_F1_3 = (overall_f1_3 - baseline_f1_3) / baseline_f1_3 if overall_f1_3 != 0 else ""
    RI_F1_4 = (overall_f1_4 - baseline_f1_4) / baseline_f1_4 if overall_f1_4 != 0 else ""
    RI_F1_5 = (overall_f1_5 - baseline_f1_5) / baseline_f1_5 if overall_f1_5 != 0 else ""

    num_of_classes_improved_to_baseline = f1_1_improved_baseline + f1_2_improved_baseline + f1_3_improved_baseline + f1_4_improved_baseline + f1_5_improved_baseline
    num_of_classes_degraded_to_baseline = 5 - num_of_classes_improved_to_baseline
    num_of_classes_no_change_to_baseline = f1_1_no_change_baseline + f1_2_no_change_baseline + f1_3_no_change_baseline + f1_4_no_change_baseline + f1_5_no_change_baseline

    ## weighted relative improvement
    FIXED_RI_F1_1 = 0 if RI_F1_1 == "" else RI_F1_1
    FIXED_RI_F1_2 = 0 if RI_F1_2 == "" else RI_F1_2
    FIXED_RI_F1_3 = 0 if RI_F1_3 == "" else RI_F1_3
    FIXED_RI_F1_4 = 0 if RI_F1_4 == "" else RI_F1_4
    FIXED_RI_F1_5 = 0 if RI_F1_5 == "" else RI_F1_5

    RI_weighted = FIXED_RI_F1_1 * class_1_weight + FIXED_RI_F1_2 * class_2_weight + FIXED_RI_F1_3 * class_3_weight + FIXED_RI_F1_4 * class_4_weight + FIXED_RI_F1_5 * class_5_weight

    ## check the RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5 and filter out the classes that improved ##
    all_classes = [FIXED_RI_F1_1, FIXED_RI_F1_2, FIXED_RI_F1_3, FIXED_RI_F1_4, FIXED_RI_F1_5]
    improved_classes = [x for x in all_classes if x > 0]
    degraded_classes = [x for x in all_classes if x < 0]

    all_majority_classes = [FIXED_RI_F1_2, FIXED_RI_F1_3, FIXED_RI_F1_4]
    improved_majority_classes = [x for x in all_majority_classes if x > 0]
    degraded_majority_classes = [x for x in all_majority_classes if x < 0]

    all_minority_classes = [FIXED_RI_F1_1, FIXED_RI_F1_5]
    improved_minority_classes = [x for x in all_minority_classes if x > 0]
    degraded_minority_classes = [x for x in all_minority_classes if x < 0]


    ## take the average of the improved classes ##
    if len(improved_classes) > 0:
        avg_RI_Improvement = sum(improved_classes) / len(improved_classes)
    else:
        avg_RI_Improvement = ""
    if len(degraded_classes) > 0:
        avg_RI_Degradation = sum(degraded_classes) / len(degraded_classes)
    else:
        avg_RI_Degradation = ""  

    if len(improved_majority_classes) > 0:
        avg_RI_Improved_Majority_Classes = sum(improved_majority_classes) / len(improved_majority_classes)
    else:
        avg_RI_Improved_Majority_Classes = ""
    if len(degraded_majority_classes) > 0:
        avg_RI_Degraded_Majority_Classes = sum(degraded_majority_classes) / len(degraded_majority_classes)
    else:
        avg_RI_Degraded_Majority_Classes = ""

    if len(improved_minority_classes) > 0:
        avg_RI_Improved_Minority_Classes = sum(improved_minority_classes) / len(improved_minority_classes)
    else:
        avg_RI_Improved_Minority_Classes = ""
    if len(degraded_minority_classes) > 0:
        avg_RI_Degraded_Minority_Classes = sum(degraded_minority_classes) / len(degraded_minority_classes)
    else:
        avg_RI_Degraded_Minority_Classes = ""      

    
    num_instances = len(y_true_all)

    ## remove np.int64 from the array
    y_index_all = [int(num) for num in y_index_all]
    y_true_all = [int(num) for num in y_true_all]
    y_pred_all = [int(num) for num in y_pred_all]

    if (overall_f1_2 > baseline_f1_2 and overall_f1_3 > baseline_f1_3 and overall_f1_4 > baseline_f1_4):
        majority_classes_Improved = "ALL MAJORITY"
    elif (overall_f1_2 > baseline_f1_2 or overall_f1_3 > baseline_f1_3 or overall_f1_4 > baseline_f1_4):
        majority_classes_Improved = "SOME MAJORITY"
    else:
        majority_classes_Improved = "NO MAJORITY"

    if (overall_f1_1 > baseline_f1_1 and overall_f1_5 > baseline_f1_5):
        minority_classes_Improved = "ALL MINORITY"
    elif (overall_f1_1 > baseline_f1_1 or overall_f1_5 > baseline_f1_5):
        minority_classes_Improved = "SOME MINORITY"
    else:
        minority_classes_Improved = "NO MINORITY"

    avg_RI_Majority_Classes = np.nanmean(all_majority_classes)
    avg_RI_Minority_Classes = np.nanmean(all_minority_classes)

    
    ### Draw Confusion Matrix ###
    # import matplotlib.pyplot as plt

    # import seaborn as sns

    # cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels(y_true_all))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels="auto", yticklabels="auto")
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # if not os.path.exists(configs.ROOT_PATH + "/results/confusion_matrix/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/confusion_matrix/" + target + "/" + model_name + "/")   
    # plt.savefig(configs.ROOT_PATH + "/results/confusion_matrix/" + target + "/" + model_name +"/" + str(K) + "_" + target + "_" + str(hyperparam) + ".png")
    # plt.close()


    # ### SAVE THE PREDICTION RESULTS WITH INDEX and ACTUAL LABELS ###
    # ### This is the Format ##
    # #### index, actual, predicted, prediction_correct, proba_1, proba_2, proba_3, proba_4, proba_5, correct_prediction
    # results_df = pd.DataFrame(list(zip(y_index_all, y_true_all, y_pred_all, np.array([proba[0] for proba in y_predict_proba_all]), np.array([proba[1] for proba in y_predict_proba_all]), np.array([proba[2] for proba in y_predict_proba_all]), np.array([proba[3] for proba in y_predict_proba_all]), np.array([proba[4] for proba in y_predict_proba_all]), (np.where(np.array(y_true_all) == np.array(y_pred_all), 1, 0)))), columns=["index", "actual_label", "predicted_label", "proba_1", "proba_2", "proba_3", "proba_4", "proba_5", "correct_prediction"])
    # if not os.path.exists(configs.ROOT_PATH + "/results/predictions/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/predictions/" + target + "/" + model_name + "/")
    # results_df.to_csv(configs.ROOT_PATH + "/results/predictions/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + ".csv", index=False)

    # Scatter plot of residuals
    # residuals = np.array(y_true_all) - np.array(y_pred_all)
    # plt.scatter(range(len(residuals)), residuals, c=residuals, cmap='coolwarm', edgecolors='k')
    # plt.axhline(y=0, color='black', linestyle='--')
    # plt.xlabel('Sample index')
    # plt.ylabel('Residuals')
    # plt.title('Residual Plot')
    # if not os.path.exists(configs.ROOT_PATH + "/results/residuals/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/residuals/" + target + "/" + model_name + "/")
    # plt.savefig(configs.ROOT_PATH + "/results/residuals/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + ".png")
    
    # ## draw histogram of the predicted probabilities for each class ##
    # plt.hist(np.array([proba[0] for proba in y_predict_proba_all]), bins=10, alpha=0.5, label='Class 1')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Predicted Probability Distribution for Class 1')
    # plt.legend(loc='upper right')
    # if not os.path.exists(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/"):
    #     os.makedirs(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/")
    # plt.savefig(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + "_class1.png")
    
    # plt.hist(np.array([proba[1] for proba in y_predict_proba_all]), bins=10, alpha=0.5, label='Class 2')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Predicted Probability Distribution for Class 2')
    # plt.legend(loc='upper right')
    # plt.savefig(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + "_class2.png")

    # plt.hist(np.array([proba[2] for proba in y_predict_proba_all]), bins=10, alpha=0.5, label='Class 3')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Predicted Probability Distribution for Class 3')
    # plt.legend(loc='upper right')
    # plt.savefig(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + "_class3.png")

    # plt.hist(np.array([proba[3] for proba in y_predict_proba_all]), bins=10, alpha=0.5, label='Class 4')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Predicted Probability Distribution for Class 4')
    # plt.legend(loc='upper right')
    # plt.savefig(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + "_class4.png")

    # plt.hist(np.array([proba[4] for proba in y_predict_proba_all]), bins=10, alpha=0.5, label='Class 5')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Frequency')
    # plt.title('Predicted Probability Distribution for Class 5')
    # plt.legend(loc='upper right')
    # plt.savefig(configs.ROOT_PATH + "/results/probability_distribution/" + target + "/" + model_name + "/" + str(K) + "_" + target + "_" + str(hyperparam) + "_class5.png")

    
    return overall_f1_weighted, overall_f1_macro, overall_precision_weighted, overall_precision_macro, overall_recall_weighted, overall_recall_macro, overall_f1_1, overall_f1_2, overall_f1_3, overall_f1_4, overall_f1_5, overall_precision_1, overall_precision_2, overall_precision_3, overall_precision_4, overall_precision_5, overall_recall_1, overall_recall_2, overall_recall_3, overall_recall_4, overall_recall_5, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, tp_4_all, tn_4_all, fp_4_all, fn_4_all, tp_5_all, tn_5_all, fp_5_all, fn_5_all, num_instances, y_index_all, y_true_all, y_pred_all, num_of_classes_improved_to_baseline, num_of_classes_degraded_to_baseline, num_of_classes_no_change_to_baseline, RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, avg_RI_Improvement, avg_RI_Degradation, RI_weighted, f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, overall_auc_1, overall_auc_2, overall_auc_3, overall_auc_4, overall_auc_5, overall_auc_weighted, overall_auc_macro, F1_weighted_improved_over_baseline, RI_F1_weighted, majority_classes_Improved, minority_classes_Improved, avg_RI_Improved_Majority_Classes, avg_RI_Degraded_Majority_Classes, avg_RI_Improved_Minority_Classes, avg_RI_Degraded_Minority_Classes, avg_RI_Majority_Classes, avg_RI_Minority_Classes



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

def dict_data_generator(model, iteration, hyperparameters, 
                        target, drop_duplicates, use_oversampling, dataset, K, 
                        fs_method, y_test_index, y_actual,y_pred, 
                        tp_1, tn_1, fp_1, fn_1, tp_2, tn_2, fp_2, fn_2, tp_3, tn_3, fp_3, fn_3, tp_4, tn_4, fp_4, fn_4, tp_5, tn_5, fp_5, fn_5,    
                        f1_1, f1_2, f1_3, f1_4, f1_5, 
                        support_1, support_2, support_3, support_4, support_5, 
                        precision_1, precision_2, precision_3, precision_4, precision_5, 
                        recall_1, recall_2, recall_3, recall_4, recall_5, 
                        n_instances, 
                        precision_weighted, recall_weighted, f1_weighted, 
                        f1_macro,  experiment, precision_macro, recall_macro, 
                        f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, 
                        num_of_classes_improved_to_baseline, num_of_classes_detegraded_to_baseline, num_of_classes_no_change_to_baseline ,
                        RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, RI_weighted, 
                        avg_RI_Improvement, avg_RI_Degradation, 
                        f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, 
                        auc_1, auc_2, auc_3, auc_4, auc_5,
                        auc_weighted, auc_macro, 
                        F1_weighted_improved_over_baseline,
                        RI_F1_weighted, 
                        majority_classes_Improved, minority_classes_Improved, 
                        avg_RI_Improved_Majority_Classes, avg_RI_Degraded_Majority_Classes, avg_RI_Majority_Classes, 
                        avg_RI_Improved_Minority_Classes, avg_RI_Degraded_Minority_Classes, avg_RI_Minority_Classes
                        ):
    csv_data_dict["model"] = model
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = str(sorted(eval(str(hyperparameters))))
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

    csv_data_dict["F1_weighted > baseline"]= F1_weighted_improved_over_baseline
    csv_data_dict["RI=(F1_weighted-baseline)/baseline"] = RI_F1_weighted

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
    csv_data_dict["RI_weighted"] = RI_weighted

    csv_data_dict["majority_classes_Improved"] = majority_classes_Improved ## Majority classes(2,3,4) improved
    csv_data_dict["minority_classes_Improved"] = minority_classes_Improved 

    csv_data_dict["avg_RI_Improved_Majority_Classes"] = avg_RI_Improved_Majority_Classes	## All Majority classes(2,3,4) improved
    csv_data_dict["avg_RI_Degraded_Majority_Classes"] = avg_RI_Degraded_Majority_Classes	
    csv_data_dict["avg_RI_Majority_Classes"] = avg_RI_Majority_Classes
    csv_data_dict["avg_RI_Improved_Minority_Classes"]= avg_RI_Improved_Minority_Classes	
    csv_data_dict["avg_RI_Degraded_Minority_Classes"] = avg_RI_Degraded_Minority_Classes
    csv_data_dict["avg_RI_Minority_Classes"] = avg_RI_Minority_Classes

    csv_data_dict["avg_RI_Improvement"] = avg_RI_Improvement
    csv_data_dict["avg_RI_Degradation"] = avg_RI_Degradation

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
                pipeline = Pipeline(steps = [ (model_name, model)])
                
                ## CONFIG 1 ## - apply over sampling
                if use_oversampling: # if use RandomOverSampling
                        # ros = RandomOverSampler(random_state=configs_DS3.RANDOM_SEED)
                        smo = SMOTE(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        # ('scaler', RobustScaler()),    
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
                    pipeline = Pipeline(steps = [(model_name, model)])
                    
                    if experiment['use_oversampling']:
                        # ros = RandomOverSampler(random_state=configs_DS3.RANDOM_SEED)
                        smo = SMOTE(random_state=configs.RANDOM_SEED)
                        pipeline = Pipeline(steps = [
                        # ('scaler', RobustScaler()),    
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
                        RI_weighted=fold_results_["RI_weighted"],

                        F1_weighted_improved_over_baseline=fold_results_["F1_weighted > baseline"],
                        RI_F1_weighted=fold_results_["RI=(F1_weighted-baseline)/baseline"],

                        majority_classes_Improved=fold_results_["majority_classes_Improved"],
                        minority_classes_Improved=fold_results_["minority_classes_Improved"],

                        avg_RI_Improved_Majority_Classes=fold_results_["avg_RI_Improved_Majority_Classes"],
                        avg_RI_Degraded_Majority_Classes=fold_results_["avg_RI_Degraded_Majority_Classes"],
                        avg_RI_Majority_Classes=fold_results_["avg_RI_Majority_Classes"],
                        avg_RI_Improved_Minority_Classes=fold_results_["avg_RI_Improved_Minority_Classes"],
                        avg_RI_Degraded_Minority_Classes=fold_results_["avg_RI_Degraded_Minority_Classes"],
                        avg_RI_Minority_Classes=fold_results_["avg_RI_Minority_Classes"],
                        
                        avg_RI_Improvement=fold_results_["avg_RI_Improvement"],
                        avg_RI_Degradation=fold_results_["avg_RI_Degradation"],
                        experiment=experiment["exp_id"])

                    dict_to_csv(configs.ROOT_PATH + "/" + configs.OUTPUT_ML_PATH, csv_data_dict1)



            
                ## aggregate the results from all the folds
                overall_f1_weighted, overall_f1_macro, overall_precision_weighted, overall_precision_macro, overall_recall_weighted, overall_recall_macro, overall_f1_1, overall_f1_2, overall_f1_3, overall_f1_4, overall_f1_5, overall_precision_1, overall_precision_2, overall_precision_3, overall_precision_4, overall_precision_5, overall_recall_1, overall_recall_2, overall_recall_3, overall_recall_4, overall_recall_5, tp_1_all, tn_1_all, fp_1_all, fn_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, tp_3_all, tn_3_all, fp_3_all, fn_3_all, tp_4_all, tn_4_all, fp_4_all, fn_4_all, tp_5_all, tn_5_all, fp_5_all, fn_5_all, num_instances, y_index_all, y_true_all, y_pred_all, num_of_classes_improved_to_baseline, num_of_classes_degraded_to_baseline, num_of_classes_no_change_to_baseline, RI_F1_1, RI_F1_2, RI_F1_3, RI_F1_4, RI_F1_5, avg_RI_Improvement, avg_RI_Degradation, RI_weighted, f1_1_improved_baseline, f1_2_improved_baseline, f1_3_improved_baseline, f1_4_improved_baseline, f1_5_improved_baseline, f1_1_no_change_baseline, f1_2_no_change_baseline, f1_3_no_change_baseline, f1_4_no_change_baseline, f1_5_no_change_baseline, overall_auc_1, overall_auc_2, overall_auc_3, overall_auc_4, overall_auc_5, overall_auc_weighted, overall_auc_macro, F1_weighted_improved_over_baseline, RI_F1_weighted, majority_classes_Improved, minority_classes_Improved, avg_RI_Improved_Majority_Classes, avg_RI_Degraded_Majority_Classes, avg_RI_Improved_Minority_Classes, avg_RI_Degraded_Minority_Classes, avg_RI_Majority_Classes, avg_RI_Minority_Classes = aggregate_results(all_fold_results, target, model_name, experiment["K %"], best_hyper_params, model) 
                
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

                    majority_classes_Improved=majority_classes_Improved,
                    minority_classes_Improved=minority_classes_Improved,
                    F1_weighted_improved_over_baseline=F1_weighted_improved_over_baseline,
                    avg_RI_Improved_Majority_Classes=avg_RI_Improved_Majority_Classes,
                    avg_RI_Degraded_Majority_Classes=avg_RI_Degraded_Majority_Classes,
                    avg_RI_Majority_Classes=avg_RI_Majority_Classes,
                    avg_RI_Improved_Minority_Classes=avg_RI_Improved_Minority_Classes,
                    avg_RI_Degraded_Minority_Classes=avg_RI_Degraded_Minority_Classes,
                    avg_RI_Minority_Classes=avg_RI_Minority_Classes,

                    RI_weighted=RI_weighted,
                    RI_F1_weighted=RI_F1_weighted,

                    avg_RI_Improvement=avg_RI_Improvement,
                    avg_RI_Degradation=avg_RI_Degradation,


                    experiment=experiment["exp_id"])
   
                dict_to_csv(configs.ROOT_PATH + "/" + configs.OUTPUT_ML_PATH, csv_data_dict2)

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

if __name__ == "__main__":
    main()