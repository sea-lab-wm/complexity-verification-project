"""
This script is for creating the experiments for the classification models.
"""

import json
import pandas as pd
import sys
# from utils import configs
from utils import configs_TASK2_DS6 as configs
sys.path.append(configs.ROOT_PATH)

import csv
import math

from featureselection.feature_selection import FeatureSelection


def experiment_generator(exp_id, target, code_comprehension_target, dynamic_epsilon, K, features, fs_method, use_oversampling):
    experiment = {
        "exp_id": "exp" + str(exp_id),
        "target": target,
        "code_comprehension_target": code_comprehension_target,
        "dynamic_epsilon": dynamic_epsilon,
        "K %": K,
        "features": features,
        "feature_selection_method": fs_method,
        "use_oversampling": use_oversampling
    }
    return experiment

def  RQ1_experiments(experiment_id, complete_df):  
    for code_comprehension_target in configs.CODE_COMPREHENSIBILITY_TARGETS:
        for epsilon in configs.DYNAMIC_EPSILON:
            
            ## filter the dataset for the target variable and dynamic epsilon value
            filtered_df = complete_df.query("target=='" + code_comprehension_target + "'" + " and " + "dynamic_epsilon==" + str(epsilon))
            warning_features = configs.WARNING_FEATURES
            target = configs.TARGET ## s2>s1 relative comprehensibility
        
            ## code features. 
            ## Note: Here "target" is the column that contains code_comprehension targets.
            ## target is the s2>s1 relative comprehensibility
            code_features = list(set(complete_df.columns) ^ set(target) ^ set(warning_features) ^ set(['target']) ^ set(['dynamic_epsilon'])) ## only code features
            
            ## code and target dataframes
            code_features_df = filtered_df[code_features]
            target_df = filtered_df[target]

            
            fs = FeatureSelection(code_features_df, target_df, code_comprehension_target, epsilon)
            mi = fs.compute_mutual_information()
            kendals = fs.compute_kendals_tau()

            ## select top k=10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% , 100% of the features
            # number_of_features = len(dataset.columns)
            for k in range(1, 11):
                k_best_features_mi = fs.select_k_best(math.floor(int(mi.shape[0] * k/10)), "mi")
                k_best_features_kendal = fs.select_k_best(math.floor(int(kendals.shape[0] * k/10)), "kendalltau")

                with open(configs.ROOT_PATH + '/' + configs.OUTPUT_PATH, "a") as file:

                    for use_oversampling in use_oversampling_config:
                                
                        #############
                        ## FS = MI ##
                        #############

                        fs_method = "MI"
                           
                        jsondump = json.dumps(experiment_generator(experiment_id, target, code_comprehension_target, epsilon, k*10, list(k_best_features_mi), fs_method, use_oversampling))
                        jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                        file.write(jsondump)
                        file.write('\n')

                        experiment_id += 1

                            

                        ###################
                        ## FS = Kendalls ##
                        ###################
                        fs_method = "Kendalls"

                        jsondump = json.dumps(experiment_generator(experiment_id, target, code_comprehension_target, epsilon, k*10, list(k_best_features_kendal), fs_method, use_oversampling))
                        jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                        file.write(jsondump)
                        file.write('\n')

                        experiment_id += 1

                        
    return experiment_id


if __name__ == "__main__":


    with open(configs.ROOT_PATH + "/" + configs.OUTPUT_PATH, "w+") as file:
        file.write("")   

    ## data loading
    df = pd.read_csv(configs.ROOT_PATH + "/" + configs.DATA_PATH)

    ## these features are not needed for the feature selection
    removed_features = configs.NOT_USEFUL_FEATURES
    complete_df = df.drop(columns=removed_features) 

    use_oversampling_config = [True, False] ### CONFIG 2 ###

    header_kendals = configs.KENDALS_HEADER
    header_mi = configs.MI_HEADER

    ## RQ1
    output_file_kendalls = configs.KENDALS_OUTPUT_FILE_NAME
    output_file_mi = configs.MI_OUTPUT_FILE_NAME


    ## write header for RQ1
    with open(configs.ROOT_PATH + '/' + output_file_kendalls, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header_kendals.keys())
        writer.writeheader()
    with open(configs.ROOT_PATH + '/' + output_file_mi, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header_mi.keys())
        writer.writeheader()

   
    experiment_id = 1 ## global variable to keep track of the experiment id

    experiment_id = RQ1_experiments(experiment_id, complete_df) 
    print("Experiments for RQ1 are generated successfully!")

