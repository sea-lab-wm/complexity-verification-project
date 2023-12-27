'''
This script is for running the experiments with regression models.

Regression models: Linear Regression, SVM, MLP, KNN, Random Forest
'''
import logging
import csv
import numpy as np
import argparse
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import os

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from scipy.stats import kendalltau

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import mean_squared_error, mean_absolute_error

## Logger
_LOG_FMT = '[%(asctime)s - %(levelname)s - %(name)s]-   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')

RANDOM_SEED=1699304546

## Num of folds for CV
folds = 5
## output file name
output_file =  "TEST_regression.csv"

## CSV data format
csv_data_dict = {
    "model": "",
    "iteration": "",
    "hyperparameters": "",
    "target": "",
    "warning_feature": "",
    "feature_set": "",
    "correlation_c": 0.0,
    "mse_c": 0.0, ## mean squared error
    "mae_c": 0.0, ## mean absolute error
    "rmse_c": 0.0, ## root mean squared error
    "correlation_cw": 0,
    "mse_cw": 0.0,
    "mae_cw": 0.0,
    "rmse_cw": 0.0,
    "diff_mse": 0.0,
    "diff_mae": 0.0,
    "diff_rmse": 0.0,
    "experiment": ""
}

ROOT_PATH = "/verification_project/"
# ROOT_PATH='/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/'

def model_initialisation(model_name, parameters):
    LOGGER.info("Launching model: " + model_name + "...")

    if model_name == "linear_regression":
        param_grid = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "positive": [True, False]
        }
        ## Pipeline requires the model name before the parameters
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()}
        model = LinearRegression()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = LinearRegression(**parameters)

    elif model_name == "svm":
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"]
        }
        ## Pipeline requires the model name before the parameters
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()}
        model = SVR()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = SVR(**parameters)

    elif model_name == "mlp":
        param_grid = {
            "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
            "activation": ["relu"],
            "solver": ["adam", "sgd"],
            "momentum": [0.9, 0.95, 0.99],
            "alpha": [0.0001],
            "learning_rate": ["constant"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "random_state": [RANDOM_SEED],
            "max_iter": [1000],
            "early_stopping": [True]
        }
        ## Pipeline requires the model name before the parameters
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()}
        model =  MLPRegressor()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = MLPRegressor(**parameters)

    elif model_name == "knn":
        param_grid = {
            "n_neighbors": [2, 3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["l1", "l2", "euclidean", "manhattan"],
        }
        ## Pipeline requires the model name before the parameters
        param_grid = {f"{model_name}__{key}":value for key, value in param_grid.items()}
        model = KNeighborsRegressor()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = KNeighborsRegressor(**parameters)

    elif model_name == "random_forest":
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
        model = RandomForestRegressor()
        if parameters:
            ## to initialize the model with the best hyperparameters model name should be removed
            parameters = {key.replace(f"{model_name}__", ""): value for key, value in parameters.items()}
            model = RandomForestRegressor(**parameters)

    else:
        LOGGER.error("Invalid model name")
        return None
    
    return model, param_grid

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

    if not os.path.exists(ROOT_PATH + "Results/regression/histograms/"):
        os.makedirs(ROOT_PATH + "Results/regression/histograms/") 

    plt.savefig(ROOT_PATH + "Results/regression/histograms/" + target + "-" + feature_set + "-" + warning_feature + "-Scaling-" +str(transformed) + ".png")
    plt.clf()
    plt.close()


def get_best_hyperparameters(kFold, model, param_grid, X_train, y_train, feature_set):
    ## GridSearchCV ##
    '''
    GridSearchCV does nested CV, all parameters are used for training on all the internal runs/splits, 
    and they are tested on the test sets. 
    The best hyperparameters are found by averaging the metric values on all the splits.
    https://scikit-learn.org/stable/_images/grid_search_cross_validation.png
    '''
    # https://datascience.stackexchange.com/questions/85546/when-using-gridsearchcv-with-regression-tree-how-to-interpret-mean-test-score
    grid = GridSearchCV(model, param_grid, cv=kFold, scoring='neg_mean_squared_error', n_jobs=-1) 
    grid.fit(X_train, y_train.values.ravel())
    
    target = y_train.columns[0] ## eg: TNPU

    warning_features = ["warnings_checker_framework", "warnings_typestate_checker", "warnings_infer", "warnings_openjml", "warning_sum"]
    warning_feature = "NO-WARNING"
    if X_train.columns[0] in warning_features:
        warning_feature = X_train.columns[0]

    ## keep the column names
    X_train_columns = [str(col) for col in X_train.columns]
    
    ## generate histograms for each target in each feature set with and without warning
    generate_histograms(X_train, target, feature_set, "False", warning_feature) 

    ## x_train distribution after Scaling before Resampling ##
    if hasattr(grid.best_estimator_, 'named_steps'):
        if hasattr(grid.best_estimator_.named_steps, 'scaler'):
            X_train_transformed = grid.best_estimator_.named_steps['scaler'].transform(X_train)
            
        ## append the column names to the resampled data
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train_columns)
        generate_histograms(X_train_transformed, target, feature_set, "True", warning_feature) ## draw the data distribution after transformation
        
    ## Best hyperparameters
    best_hyperparameters = grid.best_params_
    return best_hyperparameters

def train(model, X_train, y_train):
    ## train the model on the train split
    model.fit(X_train.values, y_train.values.ravel())
    return model

def evaluate(model, X_test, y_test):
    ## Transform the test data to Scaler ##
    X_test = model.named_steps.scaler.transform(X_test.values)   

    ## predict on the test split
    y_pred = model.predict(X_test)

    ## calculate the metrics
    mse = mean_squared_error(y_test, y_pred) ## mean squared error = np.average((y_test - y_pred) ** 2) 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    ## Kendall's tau correlation between the actual and predicted values
    corr, _ = kendalltau(y_test, y_pred, nan_policy='omit') ## nan_policy = 'omit' will ignore the Nan values
    
    ## corr = 'nan' means that all the values in the actual or predicted are the same. 
    ## Means that there is no sufficient information about ranking is provided by the predicted values to compare distributions.
    ## So the correlation is 0.0 i.e no correlation
    if math.isnan(corr): 
        corr = 0.0

    return mse, mae, rmse, corr

def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

def dict_data_generator(model_name, iteration, best_hyperparams, target, warning_feature, feature_set, 
                        corr_c, mse_c, mae_c, rmse_c, 
                        corr_cw, mse_cw, mae_cw, rmse_cw, 
                        experiment):
    csv_data_dict["model"] = model_name
    csv_data_dict["iteration"] = iteration
    csv_data_dict["hyperparameters"] = best_hyperparams
    csv_data_dict["target"] = target
    csv_data_dict["warning_feature"] = warning_feature
    csv_data_dict["feature_set"] = feature_set
    csv_data_dict["correlation_c"] = corr_c
    csv_data_dict["mse_c"] = mse_c 
    csv_data_dict["mae_c"] = mae_c ## mean absolute error = np.average(np.abs(y_test - y_pred)). Less error is better. 
    csv_data_dict["rmse_c"] = rmse_c
    csv_data_dict["correlation_cw"] = corr_cw
    csv_data_dict["mse_cw"] = mse_cw
    csv_data_dict["mae_cw"] = mae_cw
    csv_data_dict["rmse_cw"] = rmse_cw

    csv_data_dict["diff_mse"] = mse_c - mse_cw ## if the difference is positive, then the code+warning features are better than code features 
    csv_data_dict["diff_mae"] = mae_c - mae_cw
    csv_data_dict["diff_rmse"] = rmse_c - rmse_cw
    csv_data_dict["experiment"] = experiment

    return csv_data_dict

def result_aggregation(result_dict, best_hyper_params):

    ## Aggregation of the results
    overall_mse_c = np.mean([fold["mse_c"] for fold in result_dict[str(dict(best_hyper_params))]])
    overall_mae_c = np.mean([fold["mae_c"] for fold in result_dict[str(dict(best_hyper_params))]])
    overall_rmse_c = np.sqrt(overall_mse_c)

    overall_mse_cw = np.mean([fold["mse_cw"] for fold in result_dict[str(dict(best_hyper_params))]])
    overall_mae_cw = np.mean([fold["mae_cw"] for fold in result_dict[str(dict(best_hyper_params))]])
    overall_rmse_cw = np.sqrt(overall_mse_cw)

    ## https://stats.stackexchange.com/questions/494291/can-you-sum-correlation-coefficients-to-find-overall-correlation#:~:text=Adding%20different%20standardized%20covariances%20together,all%20correlations%20and%20report%20that.
    overall_corr_c = np.nanmean([fold["corr_c"] for fold in result_dict[str(dict(best_hyper_params))]]) ## nanmean ignores the nan values
    overall_corr_cw = np.nanmean([fold["corr_cw"] for fold in result_dict[str(dict(best_hyper_params))]])

    return overall_mse_c, overall_mae_c, overall_rmse_c, overall_corr_c, overall_mse_cw, overall_mae_cw, overall_rmse_cw, overall_corr_cw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression models")
    ## debug parameters
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

    ## Load the data
    feature_df = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings.csv")
    
    ## write header
    with open(ROOT_PATH + "Results/regression/" + output_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
        writer.writeheader()

    ## read the json file
    with open(ROOT_PATH + "regression/experiments.jsonl") as jsonl_file:
        experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

        model_names = ["mlp", "svm", "linear_regression", "knn", "random_forest"] 

        for model_name in model_names:
            for experiment in experiments:
                
                ## warning feature
                warning_feature = experiment["features"][0] ## eg: warnings_checker_framework
                ## feature set
                feature_set = experiment["experiment_id"].split("-")[3] ## eg: set1

                ## drop rows with missing values in the feature
                full_dataset = feature_df.dropna(subset=experiment["target"])
                
                target_y = full_dataset[experiment["target"]]
                
                ## StratifiedKFold
                kFold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
                
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

                    model_c, param_grid_c = model_initialisation(model_name, parameters="")
                    pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)]) ## StandardScaler(x) = (x-mean)/std

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
                    
                    model_cw, param_grid_cw = model_initialisation(model_name, parameters="")
                    pipeline_cw = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_cw)])
  
                    ###################
                    ## Code features ##
                    ###################
                    LOGGER.info("Best param searching for fold {} for code features...".format(ite))
                    best_hyperparams_code = get_best_hyperparameters(kFold, pipeline_c, param_grid_c,  X_train_c, y_train_c, feature_set)
                    ## remove the model name from the best hyperparameters keys
                    best_hyperparams_code = {key.replace(model_name + "__", ""):value for key, value in best_hyperparams_code.items()} 



                    ## since we are using a set, we need to convert the dict to a hashable type
                    best_hyperparams.add((frozenset(best_hyperparams_code.items())))

                    ##############################
                    ## Code + warnings features ##
                    ##############################
                    LOGGER.info("Best param searching for fold {} for code + warnings features...".format(ite))
                    best_hyperparams_code_warning = get_best_hyperparameters(kFold, pipeline_cw, param_grid_cw, X_train_cw, y_train_cw, feature_set)
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

                            model_c, _ = model_initialisation(model_name, parameters=dict((best_hyper_params)))
                            pipeline_c = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_c)])
                                
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

                            model_cw, _ = model_initialisation(model_name, parameters=dict((best_hyper_params)))
                            pipeline_cw = Pipeline(steps = [('scaler', StandardScaler()), (model_name, model_cw)])

                            ############
                            ## PART 1 ##
                            ############
                            #### code features wtih best config
                            pipeline_c = train(pipeline_c, X_train_c, y_train_c)

                            mse_c, mae_c, rmse_c, corr_c = evaluate(pipeline_c, X_test_c, y_test_c)

                            #### code + warnings features wtih best config
                            pipeline_cw = train(pipeline_cw, X_train_cw, y_train_cw)

                            mse_cw, mae_cw, rmse_cw, corr_cw = evaluate(pipeline_cw, X_test_cw, y_test_cw)

                            ## putting the results in a dictionary
                            dict_data = dict_data_generator(
                                model_name, 
                                str(ite), 
                                str(dict((best_hyper_params))),
                                experiment["target"], 
                                warning_feature, 
                                feature_set,
                                corr_c, mse_c, mae_c, rmse_c,
                                corr_cw, mse_cw, mae_cw, rmse_cw,
                                experiment["experiment_id"]
                            )

                            dict_to_csv(ROOT_PATH + "Results/regression/" + output_file, dict_data)
                            ## For each hyperparameter set, we append the list the results (in a dict)
                            result_dict[str(dict((best_hyper_params)))].append(
                                {
                                    "ite": ite,
                                    "corr_c": corr_c,
                                    "mse_c": mse_c,
                                    "mae_c": mae_c,
                                    "rmse_c": rmse_c,
                                    "corr_cw": corr_cw,
                                    "mse_cw": mse_cw,
                                    "mae_cw": mae_cw,
                                    "rmse_cw": rmse_cw
                                }
                            )

                            ite += 1

                    ## OVERALL RESULTS ACROSS ALL ITERATIONS For all configs
                    overall_mse_c, overall_mae_c, overall_rmse_c, overall_corr_c, overall_mse_cw, overall_mae_cw, overall_rmse_cw, overall_corr_cw = result_aggregation(result_dict, best_hyper_params)

                    dict_data = dict_data_generator(
                        model_name, 
                        "overall", 
                        str(dict((best_hyper_params))),
                        experiment["target"], 
                        warning_feature, 
                        feature_set,
                        overall_corr_c, overall_mse_c, overall_mae_c, overall_rmse_c,
                        overall_corr_cw, overall_mse_cw, overall_mae_cw, overall_rmse_cw,
                        experiment["experiment_id"]
                    )

                    dict_to_csv(ROOT_PATH + "Results/regression/" + output_file, dict_data)  