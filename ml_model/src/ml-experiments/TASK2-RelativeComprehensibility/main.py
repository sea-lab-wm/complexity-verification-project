## This is for computing relative code comprehensibility instead of
## absolute code comprehensibility

import pandas as pd
import numpy as np

import logging
import csv
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix, roc_auc_score, precision_score, recall_score

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')



csv_data_dict = {
    "model": "",
    "iteration": "",
    "hyperparameters_c": "",
    "target": "",
    "use_oversampling": False,
    "dynamic_epsilon": False,
    "FS_method": "",
    "K": 0,
    "y_true_index": [],
    "y_true": [],
    "y_pred": [],
    "tp_0": 0,
    "tn_0": 0,
    "fp_0": 0,
    "fn_0": 0,
    "support_0": 0,
    "tp_1": 0,
    "tn_1": 0,
    "fp_1": 0,
    "fn_1": 0,
    "support_1": 0,
    "tp_2": 0,
    "tn_2": 0,
    "fp_2": 0,
    "fn_2": 0,
    "support_2": 0,
    "#instances": 0,
    "precision_0": 0.0,
    "precision_1": 0.0,
    "precision_2": 0.0,
    "recall_0": 0.0,
    "recall_1": 0.0,
    "recall_2": 0.0,
    "f1_0_S1_more_comprehensible": 0.0,
    "f1_1_S2_more_comprehensible": 0.0,
    "f1_2_equal_comprehensible": 0.0,
    "auc_0": 0.0,
    "auc_1": 0.0,
    "auc_2": 0.0,
    "recall_weighted": 0.0,
    "precision_weighted": 0.0,
    "f1_weighted": 0.0,
    "auc_weighted": 0.0,
    "experiment": ""
}

import sys
sys.path.append('/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/relativecomprehensibility/')

from utils import configs


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
            "max_features": [None, 'sqrt', 'log2'],
            "min_impurity_decrease": [0.001, 0.01 ],
            "max_depth": [None, 10],
            # "max_depth": [30, 50],
            # "bootstrap": [True],
            # "min_samples_split": [10, 15],
            # "min_samples_leaf": [4, 5],
            "n_jobs": [-1],
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
            "solver": ["adam", "sgd"],
            "max_iter":[2000],  # Adjust based on validation
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


def train(model, X_train, y_train):
    ## train the model on the train split
    model.fit(X_train.values, y_train.values.ravel())
    return model


def evaluate(model, X_test, y_test):
    ## Transform the test data to Scaler ##
    X_test = model.named_steps.scaler.transform(X_test.values)
    
    ## predict on the test split ##
    y_pred = model.predict(X_test)

    results_dict={}

    ## if no correct predictions, return precision and recall as 0
    classifi_report = classification_report(y_test, y_pred, output_dict=True)
    prediction_report = multilabel_confusion_matrix (y_test, y_pred, labels=unique_labels(y_test))


    f1_0 = classifi_report["0"]["f1-score"]
    support_0 = classifi_report["0"]["support"]
    f1_1 = classifi_report["1"]["f1-score"]
    support_1 = classifi_report["1"]["support"]
    f1_2 = classifi_report["2"]["f1-score"]
    support_2 = classifi_report["2"]["support"]

    precision_0 = classifi_report["0"]["precision"]
    recall_0 = classifi_report["0"]["recall"]
    precision_1 = classifi_report["1"]["precision"]
    recall_1 = classifi_report["1"]["recall"]
    precision_2 = classifi_report["2"]["precision"]
    recall_2 = classifi_report["2"]["recall"]

    ## for multi-class classification 
    y_pred_prob = model.predict_proba(X_test)

    auc_0 = roc_auc_score(y_test, y_pred_prob, average=None, multi_class="ovr")[0]
    auc_1 = roc_auc_score(y_test, y_pred_prob, average=None, multi_class="ovr")[1]
    auc_2 = roc_auc_score(y_test, y_pred_prob, average=None, multi_class="ovr")[2]

    ## tp of the class 0
    tp_0 = prediction_report[0][1][1]
    tn_0 = prediction_report[0][0][0]
    fp_0 = prediction_report[0][0][1]
    fn_0 = prediction_report[0][1][0]
    ## tp of the class 1
    tp_1 = prediction_report[1][1][1]
    tn_1 = prediction_report[1][0][0]
    fp_1 = prediction_report[1][0][1]
    fn_1 = prediction_report[1][1][0]
    ## tp of the class 2
    tp_2 = prediction_report[2][1][1]
    tn_2 = prediction_report[2][0][0]
    fp_2 = prediction_report[2][0][1]
    fn_2 = prediction_report[2][1][0]

    ## compute the weighted F1 score ##
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=unique_labels(y_test))
    
    roc_auc_score_weighted = roc_auc_score(y_test, y_pred_prob, average="weighted", multi_class="ovr", labels=unique_labels(y_test))

    precision_weighted = classifi_report["weighted avg"]["precision"]
    recall_weighted = classifi_report["weighted avg"]["recall"]

    results_dict["tp_0"] = tp_0
    results_dict["tn_0"] = tn_0
    results_dict["fp_0"] = fp_0
    results_dict["fn_0"] = fn_0
    results_dict["support_0"] = support_0

    results_dict["tp_1"] = tp_1
    results_dict["tn_1"] = tn_1
    results_dict["fp_1"] = fp_1
    results_dict["fn_1"] = fn_1
    results_dict["support_1"] = support_1

    results_dict["tp_2"] = tp_2
    results_dict["tn_2"] = tn_2
    results_dict["fp_2"] = fp_2
    results_dict["fn_2"] = fn_2
    results_dict["support_2"] = support_2

    results_dict["#instances"] = len(y_test)

    results_dict["y_pred"] = y_pred
    results_dict["y_true_index"] = y_test.index
    results_dict["y_true"] = y_test.values

    results_dict["precision_0"] = precision_0
    results_dict["precision_1"] = precision_1
    results_dict["precision_2"] = precision_2

    results_dict["recall_0"] = recall_0
    results_dict["recall_1"] = recall_1
    results_dict["recall_2"] = recall_2

    results_dict["auc_0"] = auc_0
    results_dict["auc_1"] = auc_1
    results_dict["auc_2"] = auc_2

    results_dict["f1_weighted"] = f1_weighted 
    results_dict["auc_weighted"] = roc_auc_score_weighted

    results_dict["precision_weighted"] = precision_weighted
    results_dict["recall_weighted"] = recall_weighted

    results_dict["f1_0_S1_more_comprehensible"] = f1_0
    results_dict["f1_1_S2_more_comprehensible"] = f1_1
    results_dict["f1_2_equal_comprehensible"] = f1_2

    return results_dict


def dict_data_generator(model, iteration, hyperparameters_c, target, use_oversampling, dynamic_epsilon, FS_method, K, y_true_index, y_actual, y_pred, tp_0, tn_0, fp_0, fn_0, support_0, tp_1, tn_1, fp_1, fn_1, support_1, tp_2, tn_2, fp_2, fn_2, support_2, num_instances, f1_0_S1_more_comprehensible, f1_1_S2_more_comprehensible, f1_2_equal_comprehensible, f1_weighted, precision_0, precision_1, precision_2, precision_weighted, recall_0, recall_1, recall_2, recall_weighted, auc_0, auc_1, auc_2, auc_weighted, experiment):
    csv_data_dict["model"] = model
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters_c"] = hyperparameters_c
    csv_data_dict["target"] = target
    csv_data_dict["use_oversampling"] = use_oversampling
    csv_data_dict["dynamic_epsilon"] = dynamic_epsilon
    csv_data_dict["FS_method"] = FS_method
    csv_data_dict["K"] = K
    
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
    
    csv_data_dict["#instances"] = num_instances
    
    csv_data_dict["y_true_index"] = str(y_true_index).replace(", ", " ").replace("\n", " ")
    csv_data_dict["y_true"] = str(y_actual).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")
    csv_data_dict["y_pred"] = str(y_pred).replace(", ", " ").replace("\n", " ").replace("[", "").replace("]", "")

    csv_data_dict["precision_0"] = precision_0
    csv_data_dict["precision_1"] = precision_1
    csv_data_dict["precision_2"] = precision_2
    csv_data_dict["precision_weighted"] = precision_weighted

    csv_data_dict["recall_0"] = recall_0
    csv_data_dict["recall_1"] = recall_1
    csv_data_dict["recall_2"] = recall_2
    csv_data_dict["recall_weighted"] = recall_weighted

    csv_data_dict["auc_0"] = auc_0
    csv_data_dict["auc_1"] = auc_1
    csv_data_dict["auc_2"] = auc_2
    csv_data_dict["auc_weighted"] = auc_weighted

    csv_data_dict["f1_0_S1_more_comprehensible"] = f1_0_S1_more_comprehensible
    csv_data_dict["f1_1_S2_more_comprehensible"] = f1_1_S2_more_comprehensible
    csv_data_dict["f1_2_equal_comprehensible"] = f1_2_equal_comprehensible
    csv_data_dict["f1_weighted"] = f1_weighted
    
    csv_data_dict["experiment"] = experiment
    
    return csv_data_dict

def aggregate_results(results):
    """
    aggregate the results from all the folds
    """

    overall_0_f1_S1_more_comprehensible = 0
    overall_1_f1_S2_more_comprehensible = 0
    overall_2_f1_equal_comprehensible = 0
    overall_f1_weighted = 0

    overall_0_precision = 0
    overall_1_precision = 0
    overall_2_precision = 0
    overall_precision_weighted = 0

    overall_0_recall = 0
    overall_1_recall = 0
    overall_2_recall = 0
    overall_recall_weighted = 0

    tp_0_all = 0
    tn_0_all = 0
    fp_0_all = 0
    fn_0_all = 0
    support_0_all = 0

    tp_1_all = 0
    tn_1_all = 0
    fp_1_all = 0
    fn_1_all = 0
    support_1_all = 0

    tp_2_all = 0
    tn_2_all = 0
    fp_2_all = 0
    fn_2_all = 0
    support_2_all = 0

    y_index_all =[]
    y_pred_all = []
    y_true_all = []

    for key, value in results.items():
        # y_true_all.extend([int(num) for num in value["y_actual"][1:-1].split()])
        y_true_all.extend(value["y_true"])

        
    for key, value in results.items():
        y_pred_all.extend(value["y_pred"])
        y_index_all.extend(value["y_true_index"])

        ## convert [array(x)] to [x]
        # y_true_all.extend(value["y_true"].flatten())
        
        tp_0_all += value["tp_0"]
        tn_0_all += value["tn_0"]
        fp_0_all += value["fp_0"]
        fn_0_all += value["fn_0"]
        support_0_all += value["support_0"]

        tp_1_all += value["tp_1"]
        tn_1_all += value["tn_1"]
        fp_1_all += value["fp_1"]
        fn_1_all += value["fn_1"]
        support_1_all += value["support_1"]

        tp_2_all += value["tp_2"]
        tn_2_all += value["tn_2"]
        fp_2_all += value["fp_2"]
        fn_2_all += value["fn_2"]
        support_2_all += value["support_2"]


    # total_correct_predictions = [total_tn, total_fp, total_fn, total_tp] ## tn, fp, fn, tp
    # ## postive F1 score
    # overall_f1_positive = total_tp / (total_tp + 0.5*(total_fp + total_fn))
    # ## negative F1 score
    # overall_f1_negative = total_tn / (total_tn + 0.5*(total_fp + total_fn))
    # ## weighted F1 score
    # overall_f1_weighted = (overall_f1_positive * (total_tp + total_fn) + overall_f1_negative * (total_tn + total_fp)) / (total_tp + total_tn + total_fp + total_fn)

    overall_0_f1_S1_more_comprehensible = f1_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[0]
    overall_1_f1_S2_more_comprehensible = f1_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[1]
    overall_2_f1_equal_comprehensible = f1_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[2]
    
    overall_0_precision = precision_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[0]
    overall_1_precision = precision_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[1]
    overall_2_precision = precision_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[2]

    overall_0_recall = recall_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[0]
    overall_1_recall = recall_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[1]
    overall_2_recall = recall_score(y_true_all, y_pred_all, labels=[0,1,2], average=None)[2]
    
    overall_f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted")
    overall_precision_weighted = precision_score(y_true_all, y_pred_all, average="weighted")
    overall_recall_weighted = recall_score(y_true_all, y_pred_all, average="weighted")

    ## mean of the roc_auc scores
    roc_auc_score_0 = sum([value["auc_0"] for key, value in results.items()]) / len(results)
    roc_auc_score_1 = sum([value["auc_1"] for key, value in results.items()]) / len(results)
    roc_auc_score_2 = sum([value["auc_2"] for key, value in results.items()]) / len(results)
    roc_auc_score_weighted = sum([value["auc_weighted"] for key, value in results.items()]) / len(results)

    num_of_instances = len(y_true_all)
   
    ## remove np.int64 from the array
    y_index_all = [int(num) for num in y_index_all]
    y_true_all = [int(num) for num in y_true_all]
    y_pred_all = [int(num) for num in y_pred_all]
    
    # return overall results
    return overall_0_f1_S1_more_comprehensible, overall_1_f1_S2_more_comprehensible, overall_2_f1_equal_comprehensible, overall_f1_weighted, overall_0_precision, overall_1_precision, overall_2_precision, overall_precision_weighted, overall_0_recall, overall_1_recall, overall_2_recall, overall_recall_weighted, roc_auc_score_0, roc_auc_score_1, roc_auc_score_2, roc_auc_score_weighted, y_index_all, y_true_all, y_pred_all, tp_0_all, tn_0_all, fp_0_all, fn_0_all, support_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, support_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, support_2_all, num_of_instances

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def main():
    
    complete_df = pd.read_csv("/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/relativecomprehensibility/data/merged_ds6.csv")

    output_file = "relative_comprehensibility_Results_CONFIG_5.csv"

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=configs.RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=configs.RANDOM_SEED)

    ## write header
    with open(configs.ROOT_PATH + "results/" + output_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read json file
    with open(configs.ROOT_PATH + "featureselection/experiments_RQ1_filtered_config5.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
    
    models = ["svc", "mlp_classifier", "bayes_network", "knn_classifier", "logisticregression", "randomForest_classifier"]
    
    for model_name in models:
        
        for experiment in experiments:
        
            target = experiment["code_comprehension_target"]
            dynamic_epsilon = experiment["dynamic_epsilon"]
            filtered_df = complete_df.query("target == @target" + " and dynamic_epsilon ==" + str(dynamic_epsilon)) ## filter the data based on the target
            syntetic_features = experiment["features"]
            # filtered_df.to_csv("/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/results/filtered_df_AU.csv")
            ## drop developer_position, PE spec (java), PE gen from syntetic_features
            # syntetic_features = [feature for feature in syntetic_features if feature not in ["developer_position", "PE spec (java)", "PE gen"]]
            use_oversampling = experiment["use_oversampling"]

            FS_method = experiment["feature_selection_method"]

            feature_X_c = filtered_df[syntetic_features]
            K = experiment["K %"]
            
            target_y = filtered_df["(s2>s1)relative_comprehensibility"]

            ######################
            ## Best Hyperparams ##
            ######################
            best_hyperparams = set()

            ########################################
            ## BEST HYPERPARAMETERS FOR EACH FOLD ##
            ########################################
            for (fold, (train_index_c,test_index_c))  in (enumerate(outer_cv.split(feature_X_c, target_y))):
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

                model_c, param_grid_c = model_initialisation(model_name, parameters="")
                pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)])
                
                ## CONFIG 1 ## - apply over sampling
                if use_oversampling: # if use RandomOverSampling
                        ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        pipeline_c = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('ros', ros),
                        (model_name, model_c)])

                
                gridSearchCV  = GridSearchCV(pipeline_c, param_grid_c, cv=inner_cv, scoring='f1_weighted', n_jobs=-1)
                gridSearchCV.fit(X_train_c, y_train_c.values.ravel())
                best_params_c = gridSearchCV.best_params_

                # ## since we are using a set, we need to convert the dict to a hashable type
                # best_hyperparams.add((frozenset(best_params_c)))


                best_hyperparams_code = {key.replace(model_name + "__", ""):value for key, value in best_params_c.items()} 
                ## since we are using a set, we need to convert the dict to a hashable type
                best_hyperparams.add((frozenset(best_hyperparams_code.items())))
    

            all_fold_results_c = {}
            ##############################################
            ## Train and Test with best hyperparameters ##
            ##############################################
            ## Split the complete set = features + warning features in to 10 folds using outer_cv
            for best_hyper_params in best_hyperparams:
                for (fold, (train_index_c,test_index_c))  in (enumerate(outer_cv.split(feature_X_c, target_y))):

                    #############################
                    ## ONLY WITH Code features ##
                    #############################
                    ## split the Code dataset into train and test
                    X_train_c, X_test_c, y_train_c, y_test_c = (
                        pd.DataFrame(feature_X_c).iloc[train_index_c],
                        pd.DataFrame(feature_X_c).iloc[test_index_c],
                        pd.DataFrame(target_y).iloc[train_index_c],
                        pd.DataFrame(target_y).iloc[test_index_c]
                    )
                
                    model_c, _ = model_initialisation(model_name, parameters=dict(best_hyper_params))
                    pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)])

                    if experiment['use_oversampling']:
                        ros = RandomOverSampler(random_state=configs.RANDOM_SEED)
                        pipeline_c = Pipeline(steps = [
                        ('scaler', StandardScaler()),    
                        ('ros', ros),
                        (model_name, model_c)])


                    model_c = train(pipeline_c, X_train_c, y_train_c)
                    fold_results_ = evaluate(model_c, X_test_c, y_test_c)

                    all_fold_results_c[fold] = fold_results_
                                        
                    
                    ## binary
                    # tn_1_c = fold_results_["predictions"][0] #{tn, fp, fn, tp}
                    # fp_1_c = fold_results_["predictions"][1]
                    # fn_1_c = fold_results_["predictions"][2]
                    # tp_1_c = fold_results_["predictions"][3]


                    csv_data_dict1 = dict_data_generator(
                        model=model_name, 
                        iteration=fold,
                        hyperparameters_c=best_hyper_params,
                        target=target,
                        use_oversampling=use_oversampling,
                        dynamic_epsilon=dynamic_epsilon,
                        FS_method=FS_method,
                        K=K,
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
                        num_instances=fold_results_["#instances"],
                        y_true_index=fold_results_["y_true_index"],
                        y_actual=fold_results_["y_true"],
                        y_pred=fold_results_["y_pred"],
                        precision_0=fold_results_["precision_0"],
                        precision_1=fold_results_["precision_1"],
                        precision_2=fold_results_["precision_2"],
                        precision_weighted=fold_results_["precision_weighted"],
                        recall_0=fold_results_["recall_0"],
                        recall_1=fold_results_["recall_1"],
                        recall_2=fold_results_["recall_2"],
                        recall_weighted=fold_results_["recall_weighted"],
                        auc_0=fold_results_["auc_0"],
                        auc_1=fold_results_["auc_1"],
                        auc_2=fold_results_["auc_2"],
                        auc_weighted=fold_results_["auc_weighted"],
                        f1_0_S1_more_comprehensible=fold_results_["f1_0_S1_more_comprehensible"],
                        f1_1_S2_more_comprehensible=fold_results_["f1_1_S2_more_comprehensible"],
                        f1_2_equal_comprehensible=fold_results_["f1_2_equal_comprehensible"],
                        f1_weighted=fold_results_["f1_weighted"],
                        experiment=experiment["exp_id"] 
                        )

                    dict_to_csv(configs.ROOT_PATH + "results/" + output_file, csv_data_dict1)

                ## aggregate the results from all the folds  ## tn, fp, fn, tp
                overall_f1_S1_more_comprehensible, overall_f1_S2_more_comprehensible, overall_f1_equal_comprehensible, overall_f1_weighted, overall_precision_0, overall_precision_1, overall_precision_2, overall_precision_weighted, overall_recall_0, overall_recall_1, overall_recall_2, overall_recall_weighted, overall_auc_0, overall_auc_1, overall_auc_2, overall_auc_weighted, y_index_all, y_true_all, y_pred_all, tp_0_all, tn_0_all, fp_0_all, fn_0_all, support_0_all, tp_1_all, tn_1_all, fp_1_all, fn_1_all, support_1_all, tp_2_all, tn_2_all, fp_2_all, fn_2_all, support_2_all, num_of_instances = aggregate_results(all_fold_results_c)
                
                csv_data_dict2 = dict_data_generator(
                    model=model_name, 
                    iteration="overall", 
                    hyperparameters_c=best_hyper_params, 
                    target=target, 
                    use_oversampling=use_oversampling,
                    dynamic_epsilon=dynamic_epsilon,
                    FS_method=FS_method,
                    K=K,
                    tp_0=tp_0_all,
                    tn_0=tn_0_all,
                    fp_0=fp_0_all,
                    fn_0=fn_0_all,
                    support_0=support_0_all,
                    tp_1=tp_1_all,
                    tn_1=tn_1_all,
                    fp_1=fp_1_all,
                    fn_1=fn_1_all,
                    support_1=support_1_all,
                    tp_2=tp_2_all,
                    tn_2=tn_2_all,
                    fp_2=fp_2_all,
                    fn_2=fn_2_all,
                    support_2=support_2_all,
                    num_instances=num_of_instances,
                    y_true_index=y_index_all,
                    y_actual=y_true_all,
                    y_pred=y_pred_all,
                    precision_0=overall_precision_0,
                    precision_1=overall_precision_1,
                    precision_2=overall_precision_2,
                    precision_weighted=overall_precision_weighted,
                    recall_0=overall_recall_0,
                    recall_1=overall_recall_1,
                    recall_2=overall_recall_2,
                    recall_weighted=overall_recall_weighted,
                    auc_0=overall_auc_0,
                    auc_1=overall_auc_1,
                    auc_2=overall_auc_2,
                    auc_weighted=overall_auc_weighted,
                    f1_0_S1_more_comprehensible=overall_f1_S1_more_comprehensible,
                    f1_1_S2_more_comprehensible=overall_f1_S2_more_comprehensible,
                    f1_2_equal_comprehensible=overall_f1_equal_comprehensible,
                    f1_weighted=overall_f1_weighted,
                    experiment=experiment["exp_id"]
                    )

                dict_to_csv(configs.ROOT_PATH + "results/" + output_file, csv_data_dict2)

if __name__ == "__main__":
    main()