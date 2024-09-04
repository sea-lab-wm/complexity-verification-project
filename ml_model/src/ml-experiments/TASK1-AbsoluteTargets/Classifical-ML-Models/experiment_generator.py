"""
This script is for running the experiments with classification models.

Classification models: SVC, KNN, Logistic Regression, Random Forest, MLP

This requires the experiments.jsonl file to get the experiments, 
and the understandability_with_warnings.csv file to get the data.
After performing the GridSearchCV, it will find the best hyperparameters for each fold and train 
the model with the best hyperparameters.
"""

import json
import pandas as pd
from utils import configs
import csv
import math
import sys
sys.path.append('/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK1-AbsoluteTargets/Classifical-ML-Models/')

from featureselection.feature_selection import FeatureSelection


def experiment_generator(exp_id, drop_duplicates, dataset, target, K, features, warning_features, fs_method, use_oversampling):
    experiment = {
        "exp_id": "exp" + str(exp_id),
        "drop_duplicates": drop_duplicates,
        "dataset": dataset,
        "target": target,
        "K %": K,
        "features": features,
        "warning_features": warning_features, 
        "feature_selection_method": fs_method,
        "use_oversampling": use_oversampling
    }
    return experiment

def  RQ1_experiments(experiment_id):

    for drop_duplicates in drop_duplicates_config: 

        transformed_ds = configs.dataset_geneator(drop_duplicates, complete_df)
        datasets = {
            "ds_code": transformed_ds[ds_code],  
        }

        
        for target in targets:
            y = transformed_ds[target]
            for index, dataset_key in enumerate(datasets.keys()):
                dataset = datasets[dataset_key]
                fs = FeatureSelection(dataset, y, drop_duplicates, dataset_key)
                mi = fs.compute_mutual_information()
                kendals = fs.compute_kendals_tau()

                ## select top k=10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% , 100% of the features
                # number_of_features = len(dataset.columns)
                for k in range(1, 11):
                    k_best_features_mi = fs.select_k_best(math.floor(int(mi.shape[0] * k/10)), "mi")
                    k_best_features_kendal = fs.select_k_best(math.floor(int(kendals.shape[0] * k/10)), "kendalltau")

                    with open(configs.ROOT_PATH + '/' + configs.OUTPUT_RQ1_PATH, "a") as file:

                        for use_oversampling in use_oversampling_config:
                            
                            #############
                            ## FS = MI ##
                            #############

                            fs_method = "MI"
                            ## warning_sum
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_mi), [warning_features[0]], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ## 4 warning tools
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_mi), warning_features[1:], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ## all warning tools
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_mi), warning_features, fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ###################
                            ## FS = Kendalls ##
                            ###################

                            fs_method = "Kendalls"

                            ## warning_sum
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_kendal), [warning_features[0]], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ## 4 warning tools
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_kendal), warning_features[1:], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ## all warning tools
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_kendal), warning_features, fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1
    return experiment_id

def RQ2_experiments(experiment_id):
    
    for drop_duplicates in drop_duplicates_config: 
        
        transformed_ds = configs.dataset_geneator(drop_duplicates, complete_df)
        
        datasets = {
            "ds_code_sum": transformed_ds[ds_code_sum],
            "ds_code_4tools": transformed_ds[ds_code_4tools],
            "ds_all": transformed_ds[ds_all] 
        }
        

        for target in targets:
            y = transformed_ds[target]
            for index, dataset_key in enumerate(datasets.keys()):
                dataset = datasets[dataset_key]
                fs = FeatureSelection(dataset, y, drop_duplicates, dataset_key)
                mi = fs.compute_mutual_information()
                kendals = fs.compute_kendals_tau()

                ## select top k=10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% , 100% of the features
                # number_of_features = len(dataset.columns)
                for k in range(1, 11):
                    k_best_features_mi = fs.select_k_best(math.floor(int(mi.shape[0] * k/10)), "mi")
                    k_best_features_kendal = fs.select_k_best(math.floor(int(kendals.shape[0] * k/10)), "kendalltau")

                    with open(configs.ROOT_PATH + '/' + configs.OUTPUT_RQ2_PATH, "a") as file:

                        for use_oversampling in use_oversampling_config:
                            
                            #############
                            ## FS = MI ##
                            #############

                            fs_method = "MI"
                            
                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_mi), [], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

                            ###################
                            ## FS = Kendalls ##
                            ###################

                            fs_method = "Kendalls"

                            jsondump = json.dumps(experiment_generator(experiment_id, drop_duplicates, dataset_key, target, k*10, list(k_best_features_kendal), [], fs_method, use_oversampling))
                            jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                            file.write(jsondump)
                            file.write('\n')

                            experiment_id += 1

    return experiment_id    

if __name__ == "__main__":
    with open(configs.ROOT_PATH + "/" + configs.OUTPUT_RQ1_PATH, "w+") as file:
        file.write("")
    # with open(configs.ROOT_PATH + "/" + configs.OUTPUT_RQ2_PATH, "w+") as file2:
    #     file2.write("")    

    ## data loading
    df = pd.read_csv(configs.ROOT_PATH + "/" + configs.DATA_PATH)

    ## these features are not needed for the feature selection because they are not numerical
    removed_features = configs.NOT_USEFUL_FEATURES
    complete_df = df.drop(columns=removed_features) 

    targets = configs.DISCREATE_TARGETS
    continous_targets = configs.CONTINOUS_TARGETS
    warning_features=configs.WARNING_FEATURES
    
    ds_code = list(set(complete_df.columns) ^ set(targets) ^ set(continous_targets) ^ set(warning_features)) ## only code features
    ds_code_sum = ds_code + [warning_features[0]] ## only code features and warning_sum
    ds_code_4tools = ds_code + warning_features[1:] ## only code features and 4 warning tools
    ds_all = ds_code + warning_features ## all features

    drop_duplicates_config = [True, False] ### CONFIG 1 ###
    use_oversampling_config = [True, False] ### CONFIG 2 ###

    header_kendals = configs.KENDALS_HEADER
    header_mi = configs.MI_HEADER

    ## RQ1
    output_file_kendalls = configs.KENDALS_OUTPUT_FILE_NAME
    output_file_mi = configs.MI_OUTPUT_FILE_NAME

    ## RQ2
    # output_file_kendalls_rq2 = configs.KENDALS_OUTPUT_FILE_NAME_RQ2
    # output_file_mi_rq2 = configs.MI_OUTPUT_FILE_NAME_RQ2

    ## write header for RQ1
    with open(configs.ROOT_PATH + '/' + output_file_kendalls, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header_kendals.keys())
        writer.writeheader()
    with open(configs.ROOT_PATH + '/' + output_file_mi, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header_mi.keys())
        writer.writeheader()

    ## write header for RQ2
    # with open(configs.ROOT_PATH + '/' + output_file_kendalls_rq2, "w+") as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=header_kendals.keys())
    #     writer.writeheader()
    # with open(configs.ROOT_PATH + '/' + output_file_mi_rq2, "w+") as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=header_mi.keys())
    #     writer.writeheader()

    experiment_id = 1 ## global variable to keep track of the experiment id

    experiment_id = RQ1_experiments(experiment_id) 
    print("Experiments for RQ1 are generated successfully!")
    
    # RQ2_experiments(experiment_id)
    # print("Experiments for RQ2 are generated successfully!")
