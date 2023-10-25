'''
This script is for running the experiments with classification models.

Classification models: SVC, KNN, Logistic Regression, Random Forest, MLP

This requires the experiments.jsonl file to get the experiments, 
and the understandability_with_warnings.csv file to get the data.
After performing the GridSearchCV, it will find the best hyperparameters for each fold and train 
the model with the best hyperparameters.
'''

import argparse
import json
import logging
import pandas as pd
import csv
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

## ignore all warnings comes from GridSearchCV when models are not converging with the given hyperparameters
## warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, StratifiedKFold

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')

RANDOM_SEED=42

## Num of folds for CV
folds = 10
## output file name
output_file =  "test.csv"

## CSV data format
csv_data_dict = {
    "model": "",
    "iteration": "",
    "hyperparameters": "",
    "target": "",
    "tp_c": 0,
    "tn_c": 0,
    "fp_c": 0,
    "fn_c": 0,
    "n_instances_c": 0,
    "n_positives_c": 0,
    "n_negatives_c": 0,
    "tp_cw": 0,
    "tn_cw": 0,
    "fp_cw": 0,
    "fn_cw": 0,
    "n_instances_cw": 0,
    "n_positives_cw": 0,
    "n_negatives_cw": 0,
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
        ## We picked the parameters based on the following resources as believe those are the most important parameters to tune:
       
        ## https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
        param_grid = {
            "C": [1e8, 0.01, 0.1, 1, 5, 10, 15, 20],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "random_state": [RANDOM_SEED]
        }
        model = LogisticRegression() 
        if parameters:
            model = LogisticRegression(**parameters)
               
    elif model_name == "knn_classifier":
        ## https://www.kaggle.com/code/arunimsamudra/k-nn-with-hyperparameter-tuning?scriptVersionId=32640489&cellId=42
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["minowski", "euclidean", "manhattan"],
        }
        model = KNeighborsClassifier()
        if parameters:
            model = KNeighborsClassifier(**parameters)
        
    elif model_name == "randomForest_classifier":
        ## https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_features': ['sqrt', 'log2', 0.1, 0.5],
            'max_depth': [5, 10, 20, 30, 50, 100],
            'bootstrap': [True, False],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 5],
            "random_state": [RANDOM_SEED]
        }
        model = RandomForestClassifier()
        if parameters:
            model = RandomForestClassifier(**parameters)

    elif model_name == "SVC":
        ## https://medium.com/grabngoinfo/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb#:~:text=The%20most%20critical%20hyperparameters%20for%20SVM%20are%20kernel%20%2C%20C%20%2C%20and,to%20make%20it%20linearly%20separable.
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "sigmoid", "poly"],
            "degree": [2, 3, 4],
            "random_state": [RANDOM_SEED],
        }
        model = SVC()
        if parameters:
            model = SVC(**parameters)  

    elif model_name == "mlp_classifier":
        ## https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
        param_grid = {
            "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
            "activation": ["relu"],
            "solver": ["adam", "sgd"],
            "momentum": [0.9, 0.95, 0.99],
            "alpha": [0.0001],
            "learning_rate": ["constant"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "random_state": [RANDOM_SEED],
            "max_iter": [200],
            "early_stopping": [True]
        }
        model = MLPClassifier()
        if parameters:
            model = MLPClassifier(**parameters)

    elif model_name == "bayes_network":
        ## https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-naive-bayes#3
        param_grid = {
            "var_smoothing": [1e-05, 1e-09]
        }
        model = GaussianNB()
        if parameters:
            model = GaussianNB(**parameters)
    
    return model, param_grid


def get_best_hyperparameters(model_name, X_train, y_train):
    ## model initialisation
    model, param_grid = model_initialisation(model_name, parameters="")
    ## GridSearchCV ##
    '''
    GridSearchCV does nested CV, all parameters are used for training on all the internal runs/splits, 
    and they are tested on the test sets. 
    The best hyperparameters are found by averaging the metric values on all the splits.
    https://scikit-learn.org/stable/_images/grid_search_cross_validation.png
    '''
    grid = GridSearchCV(model, param_grid, cv=folds, scoring="f1", n_jobs = -1) ## F1 because it isrobust to imbalanced and balanced data (i.e., when using or not SMOTE)
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
    csv_data_dict["model"] = model_name
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = best_hyperparams
    csv_data_dict["target"] = target
    csv_data_dict["tp_c"] = tp_c
    csv_data_dict["tn_c"] = tn_c
    csv_data_dict["fp_c"] = fp_c
    csv_data_dict["fn_c"] = fn_c
    csv_data_dict["n_instances_c"] = tp_c + tn_c + fp_c + fn_c
    csv_data_dict["n_positives_c"] = tp_c + fn_c
    csv_data_dict["n_negatives_c"] = tn_c + fp_c
    csv_data_dict["tp_cw"] = tp_cw
    csv_data_dict["tn_cw"] = tn_cw
    csv_data_dict["fp_cw"] = fp_cw
    csv_data_dict["fn_cw"] = fn_cw
    csv_data_dict["n_instances_cw"] = tp_cw + tn_cw + fp_cw + fn_cw
    csv_data_dict["n_positives_cw"] = tp_cw + fn_cw
    csv_data_dict["n_negatives_cw"] = tn_cw + fp_cw
    csv_data_dict["precision_c"] = precision_c
    csv_data_dict["recall_c"] = recall_c
    csv_data_dict["f1_c"] = f1_c
    csv_data_dict["accuracy_c"] = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    csv_data_dict["precision_cw"] = precision_cw
    csv_data_dict["recall_cw"] = recall_cw
    csv_data_dict["f1_cw"] = f1_cw
    csv_data_dict["accuracy_cw"] = (tp_cw + tn_cw) / (tp_cw + tn_cw + fp_cw + fn_cw)
    csv_data_dict["auc_c"] = auc_c
    csv_data_dict["auc_cw"] = auc_cw
    csv_data_dict["diff_precision"] = precision_cw - precision_c
    csv_data_dict["diff_recall"] = recall_cw - recall_c
    csv_data_dict["diff_f1"] = f1_cw - f1_c
    csv_data_dict["diff_accuracy"] = csv_data_dict["accuracy_cw"] - csv_data_dict["accuracy_c"]
    csv_data_dict["diff_auc"] = auc_cw - auc_c
    csv_data_dict["experiment"] = experiment['experiment_id']
    csv_data_dict["use_smote"] = experiment['use_SMOTE']
    return csv_data_dict

def result_aggregation(result_dict, best_hyper_params):
    tp_c_overall = 0
    tn_c_overall = 0
    fp_c_overall = 0
    fn_c_overall = 0
    n_instances_c_overall = 0
    n_positives_c_overall = 0
    n_negatives_c_overall = 0
    tp_cw_overall = 0
    tn_cw_overall = 0
    fp_cw_overall = 0
    fn_cw_overall = 0
    n_instances_cw_overall = 0
    n_positives_cw_overall = 0
    n_negatives_cw_overall = 0
    for c_result in result_dict[str(dict(best_hyper_params))]: #len(result_dict[<config>]) = 10 (10 folds)
        tp_c_overall += c_result["tp_c"]
        tn_c_overall += c_result["tn_c"]
        fp_c_overall += c_result["fp_c"]
        fn_c_overall += c_result["fn_c"]
        n_instances_c_overall += (c_result["tp_c"] + c_result["tn_c"] + c_result["fp_c"] + c_result["fn_c"])
        n_positives_c_overall += (c_result["tp_c"] + c_result["fn_c"])
        n_negatives_c_overall += (c_result["tn_c"] + c_result["fp_c"])
        tp_cw_overall += c_result["tp_cw"]
        tn_cw_overall += c_result["tn_cw"]
        fp_cw_overall += c_result["fp_cw"]
        fn_cw_overall += c_result["fn_cw"]
        n_instances_cw_overall += (c_result["tp_cw"] + c_result["tn_cw"] + c_result["fp_cw"] + c_result["fn_cw"])
        n_positives_cw_overall += (c_result["tp_cw"] + c_result["fn_cw"])
        n_negatives_cw_overall += (c_result["tn_cw"] + c_result["fp_cw"])

    ## mean auc
    # https://stats.stackexchange.com/questions/386326/appropriate-way-to-get-cross-validated-auc#:~:text=What%20is%20the%20correct%20way,get%20the%20cross%2Dvalidated%20AUC.
    mean_auc_cw_c = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_cw_c"])
    mean_auc_c_c = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_c_c"])
                            
    precision_c_overall = tp_c_overall / (tp_c_overall + fp_c_overall)
    recall_c_overall = tp_c_overall / (tp_c_overall + fn_c_overall)
    f1_c_overall = 2 * (precision_c_overall * recall_c_overall) / (precision_c_overall + recall_c_overall)
    accuracy_overall_c = (tp_c_overall + tn_c_overall) / (tp_c_overall + tn_c_overall + fp_c_overall + fn_c_overall)

    precision_cw_overall = tp_cw_overall / (tp_cw_overall + fp_cw_overall)
    recall_cw_overall = tp_cw_overall / (tp_cw_overall + fn_cw_overall)
    f1_cw_overall = 2 * (precision_cw_overall * recall_cw_overall) / (precision_cw_overall + recall_cw_overall)
    accuracy_overall_cw = (tp_cw_overall + tn_cw_overall) / (tp_cw_overall + tn_cw_overall + fp_cw_overall + fn_cw_overall)

    return tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, n_instances_c_overall, n_positives_c_overall, n_negatives_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, n_instances_cw_overall, n_positives_cw_overall, n_negatives_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall,accuracy_overall_c ,accuracy_overall_cw, mean_auc_cw_c, mean_auc_c_c

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run classification models")
    ## debug parameters
    parser.add_argument(
        "--debug", type=int, choices=[0, 1], default=0,
        help="debug mode, output more information with debug logs" "0: disable, 1 enable")
    parser.add_argument(
        "--folds", type=int, default=5,
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

    feature_df = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings.csv")

    ## write header
    with open(ROOT_PATH + "Results/" + output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
        
        model_names = ["SVC", "knn_classifier", "logistic_regression", "randomForest_classifier", "mlp_classifier", "bayes_network"]
        
        for model_name in model_names:
            for experiment in experiments:
                ## drop rows with missing values in the feature
                full_dataset = feature_df.dropna(subset=experiment["target"])
                target_y = full_dataset[experiment["target"]]
                
                ## StratifiedKFold
                kFold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
                
                ## for code + warning features
                feature_X_cw = full_dataset[experiment["features"]]
                
                ## for code features
                ## drop the first 1 column because the first column is warning feature (control variable)
                feature_X_c = full_dataset[experiment["features"]].iloc[:, 1:]
                
                
                ## perform additional feature selection if model is not random forest
                # if model_name != "randomForest_classifier":
                #     ## perform linear floating forward feature selection
                #     selected_features = linear_floating_forward_feature_selection(feature_X_c, target_y, kFold)
                #     ## select the features
                #     feature_X_c = feature_X_c[selected_features]
                #     ## add the warning feature back
                #     selected_features = [experiment["features"][0]] + selected_features
                #     feature_X_cw = feature_X_cw[selected_features] 

                #     ## write the selected features to a jsonl file
                #     with open(ROOT_PATH + "classification/new_experiments.jsonl", "a") as jsonl_file:
                #         json.dump({"experiment_id": experiment["experiment_id"], "features": selected_features, "target": experiment["target"], "use_SMOTE" :experiment["use_SMOTE"], "model": model_name }, jsonl_file)
                #         jsonl_file.write("\n")

                ite = 1

                ## keep track of the best model for each fold
                best_hyperparams = set()

                ########################################
                ## BEST HYPERPARAMETERS FOR EACH FOLD ##
                ########################################
                ## The split will be same for both code and code + warnings features because we are using the same seed for both
                for (train_index_c, test_index_c), (train_index_cw, test_index_cw) in zip(kFold.split(feature_X_c, target_y), kFold.split(feature_X_cw, target_y)):
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

                    ## purpose of using SMOTE is to oversample the smaller class by creating synthetic samples.
                    if experiment['use_SMOTE']: # if use SMOTE
                        X_train_c, y_train_c = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_c, y_train_c.to_numpy().ravel())

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
                        X_train_cw, y_train_cw = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_cw, y_train_cw.to_numpy().ravel())

                    ###################
                    ## Code features ##
                    ###################
                    LOGGER.info("Best param searching for fold {} for code features...".format(ite))
                    best_hyperparams_code = get_best_hyperparameters(model_name, X_train_c, y_train_c)
                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams.add((frozenset(best_hyperparams_code.items())))

                    ##############################
                    ## Code + warnings features ##
                    ##############################
                    LOGGER.info("Best param searching for fold {} for code + warnings features...".format(ite))
                    best_hyperparams_code_warning = get_best_hyperparameters(model_name, X_train_cw, y_train_cw)
                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams.add((frozenset(best_hyperparams_code_warning.items())))

                    ite += 1
                
                ##############################################
                ## Train and Test with best hyperparameters ##
                ##############################################
                for best_hyper_params in best_hyperparams:
                        ite = 1

                        aucs_c_c = [] ## code features + best config
                        aucs_cw_c = []  ## code + warnings features + best config

                        result_dict = {str(dict((best_hyper_params))):[]}
                        for (train_index_c, test_index_c), (train_index_cw, test_index_cw) in zip(kFold.split(feature_X_c, target_y), kFold.split(feature_X_cw, target_y)):
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
                                X_train_c, y_train_c = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_c, y_train_c.to_numpy().ravel())

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
                                X_train_cw, y_train_cw = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_cw, y_train_cw.to_numpy().ravel())

                            ############
                            ## PART 1 ##
                            ############
                            #### code features wtih best config
                            model = train(model_name, dict((best_hyper_params)), X_train_c, y_train_c)

                            y_pred_c, tn_c, fp_c, fn_c, tp_c, precision_c, recall_c, f1_c = evaluate(model, X_test_c, y_test_c)
                            
                            fpr, tpr, thresholds = roc_curve(y_test_c, y_pred_c, pos_label=1) ## fpr - false positive rate, tpr - true positive rate
                            
                            auc_c_c = auc(fpr, tpr)
                            aucs_c_c.append(auc_c_c)
                            
                            #### code + warnings features wtih best config
                            model = train(model_name, dict((best_hyper_params)), X_train_cw, y_train_cw)
                            y_pred_cw, tn_cw, fp_cw, fn_cw, tp_cw, precision_cw, recall_cw, f1_cw = evaluate(model, X_test_cw, y_test_cw)

                            fpr, tpr, thresholds = roc_curve(y_test_cw, y_pred_cw, pos_label=1)

                            auc_cw_c = auc(fpr, tpr)
                            aucs_cw_c.append(auc_cw_c)

                            ## putting the results in a dictionary
                            dict_data = dict_data_generator(
                                model_name, 
                                str(ite), 
                                str(dict((best_hyper_params))),
                                experiment['target'], 
                                tp_c, tn_c, fp_c, fn_c, precision_c, recall_c, f1_c,
                                tp_cw, tn_cw, fp_cw, fn_cw, precision_cw, recall_cw, f1_cw, 
                                auc_c_c, auc_cw_c,  
                                experiment)
                            
                            dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)

                            ## For each hyperparameter set, we append the list the results (in a dict)
                            result_dict[str(dict((best_hyper_params)))].append({"ite": ite, "tp_c": tp_c, "tn_c": tn_c, "fp_c": fp_c, "fn_c": fn_c, 
                                                                              "tp_cw": tp_cw, "tn_cw": tn_cw, "fp_cw": fp_cw, "fn_cw": fn_cw,
                                                                              "precision_c": precision_c, "recall_c": recall_c, "f1_c": f1_c, 
                                                                              "precision_cw": precision_cw, "recall_cw": recall_cw, "f1_cw": f1_cw, 
                                                                              "aucs_c_c": aucs_c_c, "aucs_cw_c": aucs_cw_c,
                                                                              })  
                               
                            ite += 1 

                        ## OVERALL RESULTS ACROSS ALL ITERATIONS For all configs
                        tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, n_instances_c_overall, n_positives_c_overall, n_negatives_c_overall, precision_c_overall, recall_c_overall, f1_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, n_instances_cw_overall, n_positives_cw_overall, n_negatives_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall, accuracy_overall_c ,accuracy_overall_cw, mean_auc_cw_c, mean_auc_c_c = result_aggregation(result_dict, (best_hyper_params))
                        
                        dict_data = dict_data_generator(
                            model_name, 
                            "overall", 
                            str(dict((best_hyper_params))),
                            experiment['target'], 
                            tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall,
                            precision_c_overall, recall_c_overall, f1_c_overall,
                            tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, precision_cw_overall, recall_cw_overall, f1_cw_overall,
                            mean_auc_c_c, mean_auc_cw_c,
                            experiment)

                        dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)  