
import argparse
import json
import logging
from statistics import LinearRegression
import pandas as pd
import csv
import numpy as np
import warnings
from sklearn import naive_bayes

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR

## ignore all warnings comes from GridSearchCV
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')

## Num of folds for CV
folds = 10
## output file name
output_file =  "test.csv"

## CSV data format
dict_data = {
    "model": "",
    "iteration": "",
    "hyperparameters": "",
    "target": "",
    "tp_c": 0,
    "tn_c": 0,
    "fp_c": 0,
    "fn_c": 0,
    "tp_cw": 0,
    "tn_cw": 0,
    "fp_cw": 0,
    "fn_cw": 0,
    "precision_c": 0.0,
    "recall_c": 0.0,
    "f1_c": 0.0,
    "accuracy_c": 0.0,
    "precision_cw": 0.0,
    "recall_cw": 0.0,
    "f1_cw": 0.0,
    "accuracy_cw": 0.0,
    "auc_c": 0.0,
    "auc_cw": 0.0,
    "diff_precision": 0.0,
    "diff_recall": 0.0,
    "diff_f1": 0.0,
    "diff_accuracy": 0.0,
    "diff_auc": 0.0,
    "experiment": "",
    "use_smote": False
}

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"


def model_initialisation(model_name, parameters):
    LOGGER.info("Launching model: " + model_name + "...")

    if model_name == "logistic_regression":
        ## parameters for grid search
        param_grid = {
            "C": [1e8, 0.01, 0.1, 1, 5, 10, 15, 20],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "random_state": [42]
        }
        model = LogisticRegression() 
        if parameters:
            model = LogisticRegression(**parameters)
               
    elif model_name == "knn_classifier":
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
        }
        model = KNeighborsClassifier()
        if parameters:
            model = KNeighborsClassifier(**parameters)
        
    elif model_name == "randomForest_classifier":
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_features': ['sqrt', 'log2', 0.1, 0.5],
            'max_depth': [5, 6, 7, 8, 9, 10],
            'criterion': ['gini', 'entropy'],
            "random_state": [42]
        }
        model = RandomForestClassifier()
        if parameters:
            model = RandomForestClassifier(**parameters)

    elif model_name == "SVC":
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "sigmoid"],
            "random_state": [42],
        }
        model = SVC()
        if parameters:
            model = SVC(**parameters)           
    elif model_name == "mlp_classifier":
        param_grid = {
            "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [0.0001],
            "learning_rate": ["constant"],
            "learning_rate_init": [0.001],
            "random_state": [42],
            "max_iter": [200],
            "early_stopping": [True]
        }
        model = MLPClassifier()
        if parameters:
            model = MLPClassifier(**parameters)
    elif model_name == "bayes_network":
        model = naive_bayes()
        if parameters:
            model = naive_bayes(**parameters)
    elif model_name == "linear_regression":
        param_grid = {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1]
        }
        model = LinearRegression()
        if parameters:
            model = LinearRegression(**parameters)
    elif model_name == "svr":
        model = SVR()
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "sigmoid"],
            "random_state": [42],
        }
    
    return model, param_grid


def get_best_hyperparameters(model_name, X_train, y_train):
    ## model initialisation
    model, param_grid = model_initialisation(model_name, parameters="")
    ## GridSearchCV
    grid = GridSearchCV(model, param_grid, cv=folds, scoring="roc_auc", n_jobs = -1)
    ## train the model on the train split
    grid.fit(X_train, y_train)
    return grid.best_params_

def train(model_name, best_hyperparams, X_train, y_train):
    model, _ = model_initialisation(model_name, parameters=dict(best_hyperparams))
    ## train the model on the train split
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    ## predict on the test split
    y_pred = model.predict(X_test)
    ## calculate the metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return y_pred, tn, fp, fn, tp, precision, recall, f1

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def dict_data_generator(model_name, iteration, best_hyperparams, target, tp_c, tn_c, fp_c, fn_c, precision_c, recall_c, f1_c, tp_cw, tn_cw, fp_cw, fn_cw, precision_cw, recall_cw, f1_cw, auc_c, auc_cw, experiment):
    dict_data["model"] = model_name
    dict_data["iteration"] = iteration
    dict_data["hyperparameters"] = best_hyperparams
    dict_data["target"] = target
    dict_data["tp_c"] = tp_c
    dict_data["tn_c"] = tn_c
    dict_data["fp_c"] = fp_c
    dict_data["fn_c"] = fn_c
    dict_data["tp_cw"] = tp_cw
    dict_data["tn_cw"] = tn_cw
    dict_data["fp_cw"] = fp_cw
    dict_data["fn_cw"] = fn_cw
    dict_data["precision_c"] = precision_c
    dict_data["recall_c"] = recall_c
    dict_data["f1_c"] = f1_c
    dict_data["accuracy_c"] = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    dict_data["precision_cw"] = precision_cw
    dict_data["recall_cw"] = recall_cw
    dict_data["f1_cw"] = f1_cw
    dict_data["accuracy_cw"] = (tp_cw + tn_cw) / (tp_cw + tn_cw + fp_cw + fn_cw)
    dict_data["auc_c"] = auc_c
    dict_data["auc_cw"] = auc_cw
    dict_data["diff_precision"] = precision_cw - precision_c
    dict_data["diff_recall"] = recall_cw - recall_c
    dict_data["diff_f1"] = f1_cw - f1_c
    dict_data["diff_accuracy"] = dict_data["accuracy_cw"] - dict_data["accuracy_c"]
    dict_data["diff_auc"] = auc_cw - auc_c
    dict_data["experiment"] = experiment['experiment_id']
    dict_data["use_smote"] = experiment['use_SMOTE']
    return dict_data

def result_aggregation(result_dict, best_hyper_params, config_type):
    tp_c_overall = 0
    tn_c_overall = 0
    fp_c_overall = 0
    fn_c_overall = 0
    tp_cw_overall = 0
    tn_cw_overall = 0
    fp_cw_overall = 0
    fn_cw_overall = 0
    for c_result in result_dict[config_type + " - " + str(dict(best_hyper_params))]: #len(result_dict[<config>]) = 10 (10 folds)
        tp_c_overall += c_result["tp_c"]
        tn_c_overall += c_result["tn_c"]
        fp_c_overall += c_result["fp_c"]
        fn_c_overall += c_result["fn_c"]
        tp_cw_overall += c_result["tp_cw"]
        tn_cw_overall += c_result["tn_cw"]
        fp_cw_overall += c_result["fp_cw"]
        fn_cw_overall += c_result["fn_cw"]

    ## mean auc
    # https://stats.stackexchange.com/questions/386326/appropriate-way-to-get-cross-validated-auc#:~:text=What%20is%20the%20correct%20way,get%20the%20cross%2Dvalidated%20AUC.
    mean_auc_cw_c = np.mean(result_dict[config_type + " - " + str(dict(best_hyper_params))][0]["aucs_cw_c"])
    mean_auc_c_c = np.mean(result_dict[config_type + " - " + str(dict(best_hyper_params))][0]["aucs_c_c"])
    mean_auc_cw_cw = np.mean(result_dict[config_type + " - " + str(dict(best_hyper_params))][0]["aucs_cw_cw"])
    mean_auc_c_cw = np.mean(result_dict[config_type + " - " + str(dict(best_hyper_params))][0]["aucs_c_cw"])

                            
    precision_c_overall = tp_c_overall / (tp_c_overall + fp_c_overall)
    recall_c_overall = tp_c_overall / (tp_c_overall + fn_c_overall)
    f1_c_overall = 2 * (precision_c_overall * recall_c_overall) / (precision_c_overall + recall_c_overall)
    accuracy_overall_c = (tp_c_overall + tn_c_overall) / (tp_c_overall + tn_c_overall + fp_c_overall + fn_c_overall)

    precision_cw_overall = tp_cw_overall / (tp_cw_overall + fp_cw_overall)
    recall_cw_overall = tp_cw_overall / (tp_cw_overall + fn_cw_overall)
    f1_cw_overall = 2 * (precision_cw_overall * recall_cw_overall) / (precision_cw_overall + recall_cw_overall)
    accuracy_overall_cw = (tp_cw_overall + tn_cw_overall) / (tp_cw_overall + tn_cw_overall + fp_cw_overall + fn_cw_overall)

    return tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall,accuracy_overall_c ,accuracy_overall_cw, mean_auc_cw_c, mean_auc_c_c, mean_auc_cw_cw, mean_auc_c_cw

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run classification models")
    # debug parameters
    parser.add_argument(
        "--debug", type=int, choices=[0, 1], default=0,
        help="debug mode, output extra info & break all loops." "0: disable, 1 enable")
    parser.add_argument(
        "--folds", type=int, default=10,
        help="number of folds for cross validation (optional)")
    parser.add_argument(
        "--output_file", type=str, default=output_file,
        help="output file name (optional)")

    args = parser.parse_args()
    folds = args.folds
    output_file = args.output_file
    print("Processing with {} folds.".format(args.folds) if args.folds else "Processing without folds.")

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Debug mode enabled")

    df = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings.csv")

    ## write header
    with open(ROOT_PATH + "Results/" + output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writeheader()

    ## read json file
    with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
        
        model_names = ["SVC", "knn_classifier", "logistic_regression", "randomForest_classifier", "mlp_classifier"]
        
        for model_name in model_names:
            for experiment in experiments:
                ## drop rows with missing values in the feature
                full_dataset = df.dropna(subset=experiment["target"])
                target_y = full_dataset[experiment["target"]]
                
                ## StratifiedKFold
                kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                # kf = LeaveOneOut()
                
                ## for code + warning features
                feature_X_cw = full_dataset[experiment["features"]]
                
                ## for code features
                # drop the frist 1 column
                feature_X_c = full_dataset[experiment["features"]].iloc[:, 1:]
                

                ite = 1

                ## keep track of the best model for each fold
                best_hyperparams_c = set()
                best_hyperparams_cw = set()

                ########################################
                ## BEST HYPERPARAMETERS FOR EACH FOLD ##
                ########################################
                for (train_index_c, test_index_c), (train_index_cw, test_index_cw) in zip(kf.split(feature_X_c, target_y), kf.split(feature_X_cw, target_y)):
                    ###################
                    ## Code features ##
                    ###################
                    ## split the Code dataset into train and test
                    X_train_c, X_test_c, y_train_c, y_test_C = (
                        pd.DataFrame(feature_X_c).iloc[train_index_c],
                        pd.DataFrame(feature_X_c).iloc[test_index_c],
                        pd.DataFrame(target_y).iloc[train_index_c],
                        pd.DataFrame(target_y).iloc[test_index_c]
                    )

                    if experiment['use_SMOTE']: # if use SMOTE
                        X_train_c, y_train_c = SMOTE(random_state=42).fit_resample(X_train_c, y_train_c.to_numpy().ravel())

                    ##############################
                    ## Code + warnings features ##
                    ##############################
                    ## split the Code + Warning dataset into train and test
                    X_train_cw, X_test_cw, y_train_cw, y_test_cw = (
                        pd.DataFrame(feature_X_cw).iloc[train_index_cw],
                        pd.DataFrame(feature_X_cw).iloc[test_index_cw],
                        pd.DataFrame(target_y).iloc[train_index_cw],
                        pd.DataFrame(target_y).iloc[test_index_cw]
                    )


                    if experiment['use_SMOTE']:
                        X_train_cw, y_train_cw = SMOTE(random_state=42).fit_resample(X_train_cw, y_train_cw.to_numpy().ravel())

                    ###################
                    ## Code features ##
                    ###################
                    LOGGER.info("Best param searching for fold {} for code features...".format(ite))
                    best_hyperparams_code = get_best_hyperparameters(model_name, X_train_c, y_train_c)
                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams_c.add((frozenset(best_hyperparams_code.items())))

                    ##############################
                    ## Code + warnings features ##
                    ##############################
                    LOGGER.info("Best param searching for fold {} for code + warnings features...".format(ite))
                    best_hyperparams_code_warning = get_best_hyperparameters(model_name, X_train_cw, y_train_cw)
                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams_cw.add((frozenset(best_hyperparams_code_warning.items())))

                    ite += 1
                
                ##############################################
                ## Train and Test with best hyperparameters ##
                ##############################################
                for (best_hyper_params_c), (best_hyper_params_cw) in zip(best_hyperparams_c, best_hyperparams_cw):
                        ite = 1
                        
                        fprs_cw_c = [] #code+warning feature - code config
                        fprs_cw_cw = []
                        fprs_c_c = []
                        fprs_c_cw = []

                        tprs_c_c = []
                        tprs_c_cw = []
                        tprs_cw_c = []
                        tprs_cw_cw = []

                        aucs_c_c = []
                        aucs_cw_c = []  

                        aucs_c_cw = []
                        aucs_cw_cw = [] 

                        result_dict = {"c - " + str(dict(best_hyper_params_c)):[], "cw - " + str(dict(best_hyper_params_cw)):[]}
                        for (train_index_c, test_index_c), (train_index_cw, test_index_cw) in zip(kf.split(feature_X_c, target_y), kf.split(feature_X_cw, target_y)):
                            ###################
                            ## Code features ##
                            ###################
                            ## split the Code dataset into train and test
                            X_train_c, X_test_c, y_train_c, y_test_c = (
                                pd.DataFrame(feature_X_c).iloc[train_index_c],
                                pd.DataFrame(feature_X_c).iloc[test_index_c],
                                pd.DataFrame(target_y).iloc[train_index_c],
                                pd.DataFrame(target_y).iloc[test_index_c]
                            )


                            if experiment['use_SMOTE']:
                                X_train_c, y_train_c = SMOTE(random_state=42).fit_resample(X_train_c, y_train_c.to_numpy().ravel())

                            ##############################
                            ## Code + warnings features ##
                            ##############################
                            ## split the Code + Warning dataset into train and test
                            X_train_cw, X_test_cw, y_train_cw, y_test_cw = (
                                pd.DataFrame(feature_X_cw).iloc[train_index_cw],
                                pd.DataFrame(feature_X_cw).iloc[test_index_cw],
                                pd.DataFrame(target_y).iloc[train_index_cw],
                                pd.DataFrame(target_y).iloc[test_index_cw]
                            )


                            if experiment['use_SMOTE']:
                                X_train_cw, y_train_cw = SMOTE(random_state=42).fit_resample(X_train_cw, y_train_cw.to_numpy().ravel())

                            ############
                            ## PART 1 ##
                            ############
                            #### code features wtih code config
                            model = train(model_name, dict(best_hyper_params_c), X_train_c, y_train_c)
                            y_pred_c, tn_c, fp_c, fn_c, tp_c, precision_c, recall_c, f1_c = evaluate(model, X_test_c, y_test_c)
                            
                            fpr, tpr, thresholds = roc_curve(y_test_c, y_pred_c, pos_label=1)
                            
                            fprs_c_c.append(fpr)
                            tprs_c_c.append(tpr)
                            auc_c_c = auc(fpr, tpr)
                            aucs_c_c.append(auc_c_c)
                            
                            #### code + warnings features wtih code config
                            model = train(model_name, dict(best_hyper_params_c), X_train_cw, y_train_cw)
                            y_pred_cw, tn_cw, fp_cw, fn_cw, tp_cw, precision_cw, recall_cw, f1_cw = evaluate(model, X_test_cw, y_test_cw)

                            fpr, tpr, thresholds = roc_curve(y_test_cw, y_pred_cw, pos_label=1)

                            fprs_cw_c.append(fpr)
                            tprs_cw_c.append(tpr)
                            auc_cw_c = auc(fpr, tpr)
                            aucs_cw_c.append(auc_cw_c)

                        
                            dict_data = dict_data_generator(
                                model_name, 
                                str(ite), 
                                "c - "+str(dict(best_hyper_params_c)),
                                experiment['target'], 
                                tp_c, tn_c, fp_c, fn_c, precision_c, recall_c, f1_c,
                                tp_cw, tn_cw, fp_cw, fn_cw, precision_cw, recall_cw, f1_cw, 
                                auc_c_c, auc_cw_c,  
                                experiment)
                            
                            dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)

                            result_dict["c - " + str(dict(best_hyper_params_c))].append({"ite": ite, "tp_c": tp_c, "tn_c": tn_c, "fp_c": fp_c, "fn_c": fn_c, 
                                                                              "tp_cw": tp_cw, "tn_cw": tn_cw, "fp_cw": fp_cw, "fn_cw": fn_cw, 
                                                                              "precision_c": precision_c, "recall_c": recall_c, "f1_c": f1_c, 
                                                                              "precision_cw": precision_cw, "recall_cw": recall_cw, "f1_cw": f1_cw, 
                                                                              "aucs_c_c": aucs_c_c, "aucs_cw_c": aucs_cw_c,
                                                                              "aucs_c_cw": aucs_c_cw, "aucs_cw_cw": aucs_cw_cw,
                                                                              })

                            ############
                            ## PART 2 ##
                            ############
                            #### code features with code+warnings config
                            model = train(model_name, dict(best_hyper_params_cw), X_train_c, y_train_c)
                            y_pred_c, tn_c, fp_c, fn_c, tp_c, precision_c, recall_c, f1_c = evaluate(model, X_test_c, y_test_c)

                            fpr, tpr, thresholds = roc_curve(y_test_c, y_pred_c, pos_label=1)
                            fprs_c_cw.append(fpr)
                            tprs_c_cw.append(tpr)
                            auc_c_cw = auc(fpr, tpr)
                            aucs_c_cw.append(auc_c_cw)



                            #### code+warnings features with code+warnings config
                            model = train(model_name, dict(best_hyper_params_cw), X_train_cw, y_train_cw)
                            y_pred_cw, tn_cw, fp_cw, fn_cw, tp_cw, precision_cw, recall_cw, f1_cw = evaluate(model, X_test_cw, y_test_cw)

                            fpr, tpr, thresholds = roc_curve(y_test_cw, y_pred_cw, pos_label=1)
                            fprs_cw_cw.append(fpr)
                            tprs_cw_cw.append(tpr)
                            auc_cw_cw = auc(fpr, tpr)
                            aucs_cw_cw.append(auc_cw_cw)

                            dict_data = dict_data_generator(
                                model_name, 
                                str(ite), 
                                "cw - "+str(dict(best_hyper_params_cw)),
                                experiment['target'], 
                                tp_c, tn_c, fp_c, fn_c, precision_c, recall_c, f1_c,
                                tp_cw, tn_cw, fp_cw, fn_cw, precision_cw, recall_cw, f1_cw,
                                auc_c_cw, auc_cw_cw,   
                                experiment)
                            
                            dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)   


                            result_dict["cw - " + str(dict(best_hyper_params_cw))].append({"ite": ite, "tp_c": tp_c, "tn_c": tn_c, "fp_c": fp_c, "fn_c": fn_c,
                                                                                "tp_cw": tp_cw, "tn_cw": tn_cw, "fp_cw": fp_cw, "fn_cw": fn_cw, 
                                                                                "precision_c": precision_c, "recall_c": recall_c, "f1_c": f1_c, 
                                                                                "precision_cw": precision_cw, "recall_cw": recall_cw, "f1_cw": f1_cw, 
                                                                                "aucs_c_c": aucs_c_c, "aucs_cw_c": aucs_cw_c,
                                                                                "aucs_c_cw": aucs_c_cw, "aucs_cw_cw": aucs_cw_cw, 
                                                                                })    
                               
                            ite += 1 

                        ## OVERALL RESULTS ACROSS ALL ITERATIONS For code configs
                        tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall, accuracy_overall_c ,accuracy_overall_cw, mean_auc_cw_c, mean_auc_c_c, _, _ = result_aggregation(result_dict, best_hyper_params_c, "c")
                        
                        dict_data = dict_data_generator(
                            model_name, 
                            "overall", 
                            "c - "+str(dict(best_hyper_params_c)),
                            experiment['target'], 
                            tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, precision_c_overall, recall_c_overall, f1_c_overall,
                            tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall,
                            mean_auc_c_c, mean_auc_cw_c,
                            experiment)

                        dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)      


                        ## OVERALL RESULTS ACROSS ALL ITERATIONS For code+word configs
                        tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall, accuracy_overall_c , accuracy_overall_cw, _, _, mean_auc_cw_cw, mean_auc_c_cw = result_aggregation(result_dict, best_hyper_params_cw, "cw")

                        dict_data = dict_data_generator(
                            model_name, 
                            "overall", 
                            "cw - "+str(dict(best_hyper_params_cw)),
                            experiment['target'], 
                            tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, 
                            tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall,
                            mean_auc_c_cw, mean_auc_cw_cw,    
                            experiment)
                        
                        dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)   