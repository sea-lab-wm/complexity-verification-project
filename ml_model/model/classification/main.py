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
import math
from matplotlib.lines import Line2D
import seaborn as sns
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

from imblearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, auc
from sklearn.metrics import make_scorer

## ignore all warnings comes from GridSearchCV when models are not converging with the given hyperparameters
## warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SMOTENC, SMOTEN, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NeighbourhoodCleaningRule, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, CondensedNearestNeighbour, OneSidedSelection, ClusterCentroids, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, NeighbourhoodCleaningRule

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')

RANDOM_SEED=1699304546

## Num of folds for CV
folds = 10
## output file name
output_file =  "TEST_raw_1.csv"

## CSV data format
csv_data_dict = {
    "model": "",
    "iteration": "",
    "hyperparameters": "",
    "target": "",
    "warning_feature": "",
    "feature_set": "",
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
    "precision_positive_c": 0.0,
    "precision_weighted_c": 0.0,
    "precision_negative_c": 0.0,
    "recall_positive_c": 0.0,
    "recall_weighted_c": 0.0,
    "recall_negative_c": 0.0,
    "f1_positive_c": 0.0,
    "f1_weighted_c": 0.0,
    "f1_negative_c": 0.0,
    "accuracy_c": 0.0,
    "precision_positive_cw": 0.0,
    "precision_weighted_cw": 0.0,
    "precision_negative_cw": 0.0,
    "recall_positive_cw": 0.0,
    "recall_weighted_cw": 0.0,
    "recall_negative_cw": 0.0,
    "f1_positive_cw": 0.0,
    "f1_weighted_cw": 0.0,
    "f1_negative_cw": 0.0,
    "accuracy_cw": 0.0,
    "auc_c": 0.0,
    "auc_weighted_c": 0.0,
    "auc_cw": 0.0,
    "auc_weighted_cw": 0.0,
    "diff_precision_positive": 0.0,
    "diff_precision_weighted": 0.0,
    "diff_precision_negative": 0.0,
    "diff_recall_positive": 0.0,
    "diff_recall_weighted": 0.0,
    "diff_recall_negative": 0.0,
    "diff_f1_positive": 0.0,
    "diff_f1_weighted": 0.0,
    "diff_f1_negative": 0.0,
    "diff_auc": 0.0,
    "diff_auc_weighted": 0.0,
    "diff_accuracy": 0.0, # diff_accuracy --> diff_accuracy_positive
    "experiment": "",
    "use_smote": False
}

ROOT_PATH = "/home/nadeeshan/ML-Experiments/model/"

def draw_data_distributions(X_train, y_train, target, smote_status):
    '''
    Draw the distribution of the target variable
    '''
    plt.figure(figsize=(10, 6))
    plot_title = target + ' y_train distribution SMOTE-'+ smote_status
    pd.DataFrame(y_train).value_counts().plot(kind='bar', title=plot_title)
    plt.savefig(ROOT_PATH + "Results/" + plot_title + '.png')
    plt.close() 

    '''
    PCA is used to reduce the dimensionality of the data.
    '''
    pca = PCA(n_components=2)
    X_train_PCA = pca.fit_transform(pd.DataFrame(X_train).values)
    ## plot the data after PCA
    plt.figure(figsize=(10, 6))
    colors = ['blue' if label == 0 else 'green' for label in pd.DataFrame(y_train).values.ravel()]
    plt.scatter(X_train_PCA[:, 0], X_train_PCA[:, 1], c=colors)
    ## legend blue = 0, green = 1
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor='b', markersize=10), Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='g', markersize=10)])
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plot_title = target + ' X_train PCA SMOTE-'+ smote_status
    plt.title(plot_title)
    plt.savefig(ROOT_PATH + "Results/" + plot_title + ".png")
    plt.close()

def model_initialisation(model_name, parameters, smote):
    LOGGER.info("Launching model: " + model_name + "...")

    if model_name == "logisticregression":
        ## parameters for grid search
        ## We picked the parameters based on the following resources as believe those are the most important parameters to tune:
        ## https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
        param_grid = {
            "C": [1e8, 1e5, 0.01, 0.1, 1, 5, 10, 15, 20],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [4000],
            "random_state": [RANDOM_SEED]
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
            "n_neighbors": [2, 3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["l1", "l2", "euclidean", "manhattan"],
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
            "n_estimators": [10, 50, 100],
            "max_features": ['sqrt', 'log2', 0.1, 0.5],
            "max_depth": [5, 10, 20, 30, 50, 100],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 5],
            "random_state": [RANDOM_SEED]
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
            "kernel": ["rbf", "sigmoid"],
            "degree": [2, 3, 4],
            "random_state": [RANDOM_SEED],
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
            "var_smoothing": [1e-05, 1e-09]
        }
        ## Pipeline requires the model name before the parameters  
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()} 
        model = GaussianNB()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()} 
            model = GaussianNB(**parameters)
    
    return model, param_grid


def get_best_hyperparameters(model, param_grid, X_train, y_train, feature_set):
    ## model initialisation
    ## GridSearchCV ##
    '''
    GridSearchCV does nested CV, all parameters are used for training on all the internal runs/splits, 
    and they are tested on the test sets. 
    The best hyperparameters are found by averaging the metric values on all the splits.
    https://scikit-learn.org/stable/_images/grid_search_cross_validation.png
    '''

    grid = GridSearchCV(model, param_grid, cv=folds, scoring="f1_macro", refit="f1_macro",n_jobs = -1) ## F1 because it isrobust to imbalanced and balanced data (i.e., when using or not SMOTE)
    ## train the model on the train split
    grid.fit(X_train.values, y_train.values.ravel())

    target = y_train.columns[0] ## eg: PBU

    warning_features = ["warnings_checker_framework", "warnings_typestate_checker", "warnings_infer", "warnings_openjml", "warning_sum"]
    warning_feature = "NO-WARNING"
    if X_train.columns[0] in warning_features:
        warning_feature = X_train.columns[0]

    ## generate histograms for each target in each feature set with and without warning
    generate_histograms(X_train, target, feature_set, "False", warning_feature) 

    ## x_train distribution after Scaling before Resampling ##
    if hasattr(grid.best_estimator_, 'named_steps'):
        if hasattr(grid.best_estimator_.named_steps, 'scaler'):
            X_train_transformed = grid.best_estimator_.named_steps['scaler'].transform(X_train.values)
            draw_data_distributions(X_train_transformed, y_train, target, "False") ## draw the data distribution before SMOTE

    ## x_train distribution after Scaling and Resampling ##
    if hasattr(grid.best_estimator_, 'named_steps'):
        if hasattr(grid.best_estimator_.named_steps, 'scaler'):
            X_resampled = grid.best_estimator_.named_steps['scaler'].transform(X_train.values)
            y_resampled = y_train
            
        if hasattr(grid.best_estimator_.named_steps, 'smote'):
            X_resampled, y_resampled = grid.best_estimator_.named_steps['smote'].fit_resample(X_resampled, y_resampled)
        
        generate_histograms(pd.DataFrame(X_resampled), target, feature_set, "True", warning_feature) ## draw the data distribution after SMOTE
        draw_data_distributions(X_resampled, y_resampled, target, "True") ## draw the data distribution after SMOTE

    return grid.best_params_

def generate_histograms(X_train, target, feature_set, transformed, warning_feature):
    '''
    Generate histograms for each feature in the training data
    '''
    plt.figure(figsize=(15, 15))

    bins =  "fd" ## Freedman-Diaconis rule. which is less sensitive to outliers in the data and better for skewed data.
    for feature in X_train.columns:
        plt.hist(X_train[feature], bins, alpha=0.9, label=feature, 
                         linewidth=1, edgecolor='black')
        # # density probability distribution
        # sns.kdeplot(X_train[feature], color='black', linewidth=1, alpha=0.9)
        plt.legend(loc='upper right', ncol=3)
             
    plt.ylabel('Count')
    plt.xlabel('Bins')
    plt.savefig(ROOT_PATH + "results/classification/histograms/" + target + "-" + feature_set + "-" + warning_feature + "-Scaling-" +str(transformed) + ".png")
    plt.clf()
    plt.close()

def train(model, X_train, y_train):
    ## train the model on the train split
    model.fit(X_train.values, y_train.values.ravel())
    return model

def evaluate(model, X_test, y_test):
    ## Transform the test data to Scaler ##
    X_test = model.named_steps.scaler.transform(X_test.values)
    
    ## predict on the test split
    y_pred = model.predict(X_test)

    ## calculate the metrics for positive class ##
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    
    precision_positive = precision_score(y_test, y_pred) ## tp / (tp + fp)
    recall_positive = recall_score(y_test, y_pred) ## tp / (tp + fn)
    f1_positive = f1_score(y_test, y_pred) ## 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
    
    if tp == 0 and fp == 0:
        precision_positive = 0
    if tp == 0 and fn == 0:
        recall_positive = 0
    if precision_positive == 0 and recall_positive == 0:
        f1_positive = 0

    ## calculate the metrics for negative class ##
    ## Negative predictive value https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values#Negative_predictive_value_(NPV)
    precision_negative = tn / (tn + fn) 
    recall_negative = tn / (tn + fp)
    f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)
    
    if tn == 0 and fn == 0:
        precision_negative = 0
    if tn == 0 and fp == 0:
        recall_negative = 0
    if precision_negative == 0 and recall_negative == 0:
        f1_negative = 0

    n_positives = tp + fn ## num positive instances
    n_negatives = tn + fp ## num negative instances

    ## calculate the weighted metrics ## 
    precision_weighted = (precision_positive * n_positives + precision_negative * n_negatives) / (n_positives + n_negatives)
    recall_weighted = (recall_positive * n_positives + recall_negative * n_negatives) / (n_positives + n_negatives)
    ## https://www.v7labs.com/blog/f1-score-guideR  
    f1_weighted = (f1_positive * n_positives + f1_negative * n_negatives) / (n_positives + n_negatives)

    ## calculate the auc
    auc_value = roc_auc_score(y_test, y_pred)
    auc_weighted = roc_auc_score(y_test, y_pred, average='weighted') ## weighted

    ## accuracy
    accuracy = accuracy_score(y_test, y_pred) ## (tp + tn) / (tp + tn + fp + fn)

    return y_pred, tn, fp, fn, tp, precision_positive,  precision_weighted, precision_negative,  recall_positive, recall_weighted, recall_negative, f1_positive,  f1_weighted, f1_negative, auc_value, auc_weighted, accuracy

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def dict_data_generator(model_name, iteration, best_hyperparams, target, warning_feature, feature_set, 
                        tp_c, tn_c, fp_c, fn_c, 
                        precision_positive_c, precision_weighted_c,
                        precision_negative_c,
                        recall_positive_c, recall_weighted_c,
                        recall_negative_c,
                        f1_positive_c, f1_weighted_c, 
                        f1_negative_c,
                        tp_cw, tn_cw, fp_cw, fn_cw, 
                        precision_positive_cw, precision_weighted_cw, 
                        precision_negative_cw,
                        recall_positive_cw, recall_weighted_cw,
                        recall_negative_cw,
                        f1_positive_cw, f1_weighted_cw, 
                        f1_negative_cw,
                        auc_c, auc_weighted_c, auc_cw, auc_weighted_cw,
                        accuracy_c,
                        accuracy_cw,
                        experiment):
    csv_data_dict["model"] = model_name
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = best_hyperparams
    csv_data_dict["target"] = target
    csv_data_dict["warning_feature"] = warning_feature
    csv_data_dict["feature_set"] = feature_set
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

    csv_data_dict["precision_positive_c"] = precision_positive_c
    csv_data_dict["precision_weighted_c"] = precision_weighted_c
    csv_data_dict["precision_negative_c"] = precision_negative_c

    csv_data_dict["recall_positive_c"] = recall_positive_c
    csv_data_dict["recall_weighted_c"] = recall_weighted_c
    csv_data_dict["recall_negative_c"] = recall_negative_c

    csv_data_dict["f1_positive_c"] = f1_positive_c
    csv_data_dict["f1_weighted_c"] = f1_weighted_c
    csv_data_dict["f1_negative_c"] = f1_negative_c

    csv_data_dict["accuracy_c"] = accuracy_c ## (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    
    csv_data_dict["precision_positive_cw"] = precision_positive_cw
    csv_data_dict["precision_weighted_cw"] = precision_weighted_cw
    csv_data_dict["precision_negative_cw"] = precision_negative_cw

    csv_data_dict["recall_positive_cw"] = recall_positive_cw
    csv_data_dict["recall_weighted_cw"] = recall_weighted_cw
    csv_data_dict["recall_negative_cw"] = recall_negative_cw


    csv_data_dict["f1_positive_cw"] = f1_positive_cw
    csv_data_dict["f1_weighted_cw"] = f1_weighted_cw
    csv_data_dict["f1_negative_cw"] = f1_negative_cw

    csv_data_dict["accuracy_cw"] = accuracy_cw ## (tp_cw + tn_cw) / (tp_cw + tn_cw + fp_cw + fn_cw)
    

    csv_data_dict["auc_c"] = auc_c
    csv_data_dict["auc_weighted_c"] = auc_weighted_c
    csv_data_dict["auc_cw"] = auc_cw
    csv_data_dict["auc_weighted_cw"] = auc_weighted_cw

    csv_data_dict["diff_precision_positive"] = precision_positive_cw - precision_positive_c
    csv_data_dict["diff_precision_weighted"] = precision_weighted_cw - precision_weighted_c
    csv_data_dict["diff_precision_negative"] = precision_negative_cw - precision_negative_c
    
    csv_data_dict["diff_recall_positive"] = recall_positive_cw - recall_positive_c
    csv_data_dict["diff_recall_weighted"] = recall_weighted_cw - recall_weighted_c
    csv_data_dict["diff_recall_negative"] = recall_negative_cw - recall_negative_c
    
    csv_data_dict["diff_f1_positive"] = f1_positive_cw - f1_positive_c
    csv_data_dict["diff_f1_weighted"] = f1_weighted_cw - f1_weighted_c
    csv_data_dict["diff_f1_negative"] = f1_negative_cw - f1_negative_c

    csv_data_dict["diff_auc"] = auc_cw - auc_c
    csv_data_dict["diff_auc_weighted"] = auc_weighted_cw - auc_weighted_c

    csv_data_dict["diff_accuracy"] = csv_data_dict["accuracy_cw"] - csv_data_dict["accuracy_c"]

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
    mean_auc_cw = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_cw"])
    mean_auc_weighted_cw = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_weighted_cw"])
    mean_auc_c = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_c"])
    mean_auc_weighted_c = np.mean(result_dict[str(dict(best_hyper_params))][0]["aucs_weighted_c"])

    n_positives_c_overall = tp_c_overall + fn_c_overall
    n_negatives_c_overall = tn_c_overall + fp_c_overall

    precision_positive_c_overall = tp_c_overall / (tp_c_overall + fp_c_overall)
    if tp_c_overall == 0 and fp_c_overall == 0:
        precision_positive_c_overall = 0
    precision_negative_c_overall = tn_c_overall / (tn_c_overall + fn_c_overall)
    if tn_c_overall == 0 and fn_c_overall == 0:
        precision_negative_c_overall = 0
    precision_weighted_c_overall = (precision_positive_c_overall * n_positives_c_overall + precision_negative_c_overall * n_negatives_c_overall) / (n_positives_c_overall + n_negatives_c_overall)
    
    recall_positive_c_overall = tp_c_overall / (tp_c_overall + fn_c_overall)
    if tp_c_overall == 0 and fn_c_overall == 0:
        recall_positive_c_overall = 0
    recall_negative_c_overall = tn_c_overall / (tn_c_overall + fp_c_overall)
    if tn_c_overall == 0 and fp_c_overall == 0:
        recall_negative_c_overall = 0
    recall_weighted_c_overall = (recall_positive_c_overall * n_positives_c_overall + recall_negative_c_overall * n_negatives_c_overall) / (n_positives_c_overall + n_negatives_c_overall)
         
    
    f1_positive_c_overall = 2 * (precision_positive_c_overall * recall_positive_c_overall) / (precision_positive_c_overall + recall_positive_c_overall)
    if precision_positive_c_overall == 0 and recall_positive_c_overall == 0:
        f1_positive_c_overall = 0
    f1_negative_c_overall = 2 * (precision_negative_c_overall * recall_negative_c_overall) / (precision_negative_c_overall + recall_negative_c_overall)
    if precision_negative_c_overall == 0 and recall_negative_c_overall == 0:
        f1_negative_c_overall = 0
    f1_weighted_c_overall = (f1_positive_c_overall * n_positives_c_overall + f1_negative_c_overall * n_negatives_c_overall) / (n_positives_c_overall + n_negatives_c_overall)                

    accuracy_overall_c = (tp_c_overall + tn_c_overall) / (tp_c_overall + tn_c_overall + fp_c_overall + fn_c_overall)
    
    n_positives_cw_overall = tp_cw_overall + fn_cw_overall
    n_negatives_cw_overall = tn_cw_overall + fp_cw_overall

    precision_positive_cw_overall = tp_cw_overall / (tp_cw_overall + fp_cw_overall)
    if tp_cw_overall == 0 and fp_cw_overall == 0:
        precision_positive_cw_overall = 0
    precision_negative_cw_overall = tn_cw_overall / (tn_cw_overall + fn_cw_overall)
    if tn_cw_overall == 0 and fn_cw_overall == 0:
        precision_negative_cw_overall = 0
    precision_weighted_cw_overall = (precision_positive_cw_overall * n_positives_cw_overall + precision_negative_cw_overall * n_negatives_cw_overall) / (n_positives_cw_overall + n_negatives_cw_overall)

    recall_positive_cw_overall = tp_cw_overall / (tp_cw_overall + fn_cw_overall)
    if tp_cw_overall == 0 and fn_cw_overall == 0:
        recall_positive_cw_overall = 0
    recall_negative_cw_overall = tn_cw_overall / (tn_cw_overall + fp_cw_overall)
    if tn_cw_overall == 0 and fp_cw_overall == 0:
        recall_negative_cw_overall = 0
    recall_weighted_cw_overall = (recall_positive_cw_overall * n_positives_cw_overall + recall_negative_cw_overall * n_negatives_cw_overall) / (n_positives_cw_overall + n_negatives_cw_overall)
    
    f1_positive_cw_overall = 2 * (precision_positive_cw_overall * recall_positive_cw_overall) / (precision_positive_cw_overall + recall_positive_cw_overall)
    if precision_positive_cw_overall == 0 and recall_positive_cw_overall == 0:
        f1_positive_cw_overall = 0
    f1_negative_cw_overall = 2 * (precision_negative_cw_overall * recall_negative_cw_overall) / (precision_negative_cw_overall + recall_negative_cw_overall)
    if precision_negative_cw_overall == 0 and recall_negative_cw_overall == 0:
        f1_negative_cw_overall = 0
    f1_weighted_cw_overall = (f1_positive_cw_overall * n_positives_cw_overall + f1_negative_cw_overall * n_negatives_cw_overall) / (n_positives_cw_overall + n_negatives_cw_overall)   

    accuracy_overall_cw = (tp_cw_overall + tn_cw_overall) / (tp_cw_overall + tn_cw_overall + fp_cw_overall + fn_cw_overall)
    
    return tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, n_instances_c_overall, n_positives_c_overall, n_negatives_c_overall, precision_positive_c_overall, precision_weighted_c_overall, precision_negative_c_overall, recall_positive_c_overall, recall_weighted_c_overall, recall_negative_c_overall, f1_positive_c_overall, f1_weighted_c_overall, f1_negative_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, n_instances_cw_overall, n_positives_cw_overall, n_negatives_cw_overall, precision_positive_cw_overall, precision_weighted_cw_overall, precision_negative_cw_overall, recall_positive_cw_overall, recall_weighted_cw_overall, recall_negative_cw_overall, f1_positive_cw_overall, f1_weighted_cw_overall, f1_negative_cw_overall, mean_auc_cw, mean_auc_weighted_cw, mean_auc_c, mean_auc_weighted_c, accuracy_overall_c ,accuracy_overall_cw

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
    feature_df_PBU = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_PBU.csv")
    feature_df_BD50 = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_BD50.csv")
    feature_df_ABU50 = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_ABU50.csv")

    feature_df_PBU_set3 = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_PBU_set3.csv")
    feature_df_BD50_set3 = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_BD50_set3.csv")
    feature_df_ABU50_set3 = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_ABU50_set3.csv")
    ## write header
    with open(ROOT_PATH + "Results/" + output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
        
        # model_names = ["logisticregression", "knn_classifier", "svc", "mlp_classifier", "bayes_network", "randomForest_classifier"]
        model_names = ["svc"]
        for model_name in model_names:
            for experiment in experiments:

                ## warning feature
                warning_feature = experiment["features"][0] ## eg: warnings_checker_framework
                ## feature set
                feature_set = experiment["experiment_id"].split("-")[3] ## eg: set1

                ## feature data selection
                if experiment["target"] == "PBU":
                    feature_df = feature_df_PBU
                    if feature_set == "set3":
                        feature_df = feature_df_PBU_set3
                elif experiment["target"] == "BD50":
                    feature_df = feature_df_BD50
                    if feature_set == "set3":
                        feature_df = feature_df_BD50_set3
                elif experiment["target"] == "ABU50":
                    feature_df = feature_df_ABU50
                    if feature_set == "set3":
                        feature_df = feature_df_ABU50_set3 
                LOGGER.info("Loading data for {}...".format(experiment["target"]))       

                

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

                    model_c, param_grid_c = model_initialisation(model_name, parameters="", smote=experiment['use_SMOTE'])
                    pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)])
                    # pipeline = Pipeline(steps = [(model_name, model)])
                    if experiment['use_SMOTE']: # if use SMOTE
                        smote = SMOTE(random_state=RANDOM_SEED)
                        pipeline_c = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('smote', smote),
                        (model_name, model_c)])
                   

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


                    
                    model_cw, param_grid_cw = model_initialisation(model_name, parameters="", smote=experiment['use_SMOTE'])
                    pipeline_cw = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_cw)])
                    # pipeline = Pipeline(steps = [(model_name, model)])
                    if experiment['use_SMOTE']:
                        smote = SMOTE(random_state=RANDOM_SEED)
                        pipeline_cw = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('smote', smote),
                        (model_name, model_cw)])

                       
                    ###################
                    ## Code features ##
                    ###################
                    LOGGER.info("Best param searching for fold {} for code features...".format(ite))
                    best_hyperparams_code = get_best_hyperparameters(pipeline_c, param_grid_c,  X_train_c, y_train_c, feature_set)
                    ## remove the model name from the best hyperparameters keys
                    best_hyperparams_code = {key.replace(model_name + "__", ""):value for key, value in best_hyperparams_code.items()} 



                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams.add((frozenset(best_hyperparams_code.items())))

                    ##############################
                    ## Code + warnings features ##
                    ##############################
                    LOGGER.info("Best param searching for fold {} for code + warnings features...".format(ite))
                    best_hyperparams_code_warning = get_best_hyperparameters(pipeline_cw, param_grid_cw, X_train_cw, y_train_cw, feature_set)
                    ## split from __ and keep the parameters
                    best_hyperparams_code_warning = {key.replace(model_name + "__", ""):value for key, value in best_hyperparams_code_warning.items()}
                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams.add((frozenset(best_hyperparams_code_warning.items())))

                    ite += 1
                
                ##############################################
                ## Train and Test with best hyperparameters ##
                ##############################################
                for best_hyper_params in best_hyperparams:
                        ite = 1

                        aucs_c = [] ## code features + best config
                        aucs_cw = []  ## code + warnings features + best config
                        aucs_weighted_c = []
                        aucs_weighted_cw = []

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
                                pd.DataFrame(target_y).iloc[test_index_c])

                            model_c, _ = model_initialisation(model_name, parameters=dict((best_hyper_params)), smote=experiment['use_SMOTE'])
                            pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)])
                            # pipeline = Pipeline(steps = [(model_name, model)])
                            if experiment['use_SMOTE']:
                                smote = SMOTE(random_state=RANDOM_SEED)
                                pipeline_c = Pipeline(steps = [
                                ('scaler', StandardScaler()),    
                                ('smote', smote),
                                (model_name, model_c)])

                                
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

                            model_cw, _ = model_initialisation(model_name, parameters=dict((best_hyper_params)), smote=experiment['use_SMOTE'])
                            pipeline_cw = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_cw)])
                            # pipeline = Pipeline(steps = [(model_name, model)])
                            if experiment['use_SMOTE']:

                                smote = SMOTE(random_state=RANDOM_SEED)
                                pipeline_cw = Pipeline(steps = [
                                ('scaler', StandardScaler()),    
                                ('smote', smote),
                                (model_name, model_cw)])
 
                            ############
                            ## PART 1 ##
                            ############
                            #### code features wtih best config
                            pipeline_c = train(pipeline_c, X_train_c, y_train_c)

                            y_pred_c, tn_c, fp_c, fn_c, tp_c, precision_positive_c,  precision_weighted_c, precision_negative_c, recall_positive_c, recall_weighted_c, recall_negative_c, f1_positive_c,  f1_weighted_c, f1_negative_c, auc_c, auc_weighted_c, accuracy_c = evaluate(pipeline_c, X_test_c, y_test_c)
                            
                            aucs_c.append(auc_c)
                            aucs_weighted_c.append(auc_weighted_c)
                            
                            
                            
                            #### code + warnings features wtih best config
                            pipeline_cw = train(pipeline_cw, X_train_cw, y_train_cw)
                            
                            y_pred_cw, tn_cw, fp_cw, fn_cw, tp_cw, precision_positive_cw,  precision_weighted_cw, precision_negative_cw, recall_positive_cw, recall_weighted_cw, recall_negative_cw, f1_positive_cw,  f1_weighted_cw, f1_negative_cw, auc_cw, auc_weighted_cw, accuracy_cw = evaluate(pipeline_cw, X_test_cw, y_test_cw)
                            
                            aucs_cw.append(auc_cw)
                            aucs_weighted_cw.append(auc_weighted_cw)

                            
                            
                            ## putting the results in a dictionary
                            dict_data = dict_data_generator(
                                model_name, 
                                str(ite), 
                                str(dict((best_hyper_params))),
                                experiment['target'], 
                                warning_feature,
                                feature_set,
                                tp_c, tn_c, fp_c, fn_c, 
                                precision_positive_c, precision_weighted_c,
                                precision_negative_c,  
                                recall_positive_c, recall_weighted_c, 
                                recall_negative_c,
                                f1_positive_c, f1_weighted_c,
                                f1_negative_c,
                                tp_cw, tn_cw, fp_cw, fn_cw, 
                                precision_positive_cw, precision_weighted_cw,
                                precision_negative_cw,
                                recall_positive_cw, recall_weighted_cw,
                                recall_negative_cw, 
                                f1_positive_cw, f1_weighted_cw,
                                f1_negative_cw, 
                                auc_c, auc_weighted_c,
                                auc_cw, auc_weighted_cw, 
                                accuracy_c, 
                                accuracy_cw, 
                                experiment)
                            
                            dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)

                            ## For each hyperparameter set, we append the list the results (in a dict)
                            result_dict[str(dict((best_hyper_params)))].append({"ite": ite, "tp_c": tp_c, "tn_c": tn_c, "fp_c": fp_c, "fn_c": fn_c, 
                                                                              "tp_cw": tp_cw, "tn_cw": tn_cw, "fp_cw": fp_cw, "fn_cw": fn_cw,
                                                                              "precision_positive_c": precision_positive_c, "precision_weighted_c": precision_weighted_c,
                                                                              "precision_negative_c": precision_negative_c, 
                                                                              "recall_positive_c": recall_positive_c, "recall_weighted_c": recall_weighted_c,
                                                                              "recall_negative_c": recall_negative_c,
                                                                              "f1_positive_c": f1_positive_c, "f1_weighted_c": f1_weighted_c,
                                                                              "f1_negative_c": f1_negative_c, 
                                                                              "precision_positive_cw": precision_positive_cw, "precision_weighted_cw": precision_weighted_cw,
                                                                              "precision_negative_cw": precision_negative_cw, 
                                                                              "recall_positive_cw": recall_positive_cw, "recall_weighted_cw": recall_weighted_cw,
                                                                              "recall_negative_cw": recall_negative_cw,
                                                                              "f1_positive_cw": f1_positive_cw, "f1_weighted_cw": f1_weighted_cw,
                                                                              "f1_negative_cw": f1_negative_cw,
                                                                              "aucs_c": aucs_c,
                                                                              "aucs_weighted_c": aucs_weighted_c,  
                                                                              "aucs_cw": aucs_cw,
                                                                              "aucs_weighted_cw": aucs_weighted_cw,
                                                                              "accuracy_c": accuracy_c,
                                                                              "accuracy_cw": accuracy_cw
                                                                              })  
                               
                            ite += 1 

                        ## OVERALL RESULTS ACROSS ALL ITERATIONS For all configs
                        tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall, n_instances_c_overall, n_positives_c_overall, n_negatives_c_overall, precision_positive_c_overall, precision_weighted_c_overall, precision_negative_c_overall, recall_positive_c_overall, recall_weighted_c_overall, recall_negative_c_overall, f1_positive_c_overall, f1_weighted_c_overall, f1_negative_c_overall, tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, n_instances_cw_overall, n_positives_cw_overall, n_negatives_cw_overall, precision_positive_cw_overall, precision_weighted_cw_overall, precision_negative_cw_overall, recall_positive_cw_overall, recall_weighted_cw_overall, recall_negative_cw_overall, f1_positive_cw_overall, f1_weighted_cw_overall, f1_negative_cw_overall, mean_auc_cw, mean_auc_weighted_cw, mean_auc_c, mean_auc_weighted_c, accuracy_overall_c ,accuracy_overall_cw = result_aggregation(result_dict, (best_hyper_params))
                        
                        dict_data = dict_data_generator(
                            model_name, 
                            "overall", 
                            str(dict((best_hyper_params))),
                            experiment['target'], 
                            warning_feature,
                            feature_set,
                            tp_c_overall, tn_c_overall, fp_c_overall, fn_c_overall,
                            precision_positive_c_overall, precision_weighted_c_overall,
                            precision_negative_c_overall, 
                            recall_positive_c_overall, recall_weighted_c_overall,
                            recall_negative_c_overall, 
                            f1_positive_c_overall, f1_weighted_c_overall,
                            f1_negative_c_overall, 
                            tp_cw_overall, tn_cw_overall, fp_cw_overall, fn_cw_overall, 
                            precision_positive_cw_overall, precision_weighted_cw_overall,
                            precision_negative_cw_overall, 
                            recall_positive_cw_overall, recall_weighted_cw_overall,
                            recall_negative_cw_overall, 
                            f1_positive_cw_overall, f1_weighted_cw_overall,
                            f1_negative_cw_overall, 
                            mean_auc_c, mean_auc_weighted_c,
                            mean_auc_cw, mean_auc_weighted_cw,
                            accuracy_c,
                            accuracy_cw,
                            experiment)

                        dict_to_csv(ROOT_PATH + "Results/" + output_file, dict_data)  