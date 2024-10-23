#############
#### Analysis of the results of the both Task1 and Task2
#############

import pandas as pd
import numpy as np
import csv
import os

ROOT_PATH="/home/nadeeshan/VPro2/complexity-verification-project/ml_model/src/ml-experiments"

DS3_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/DS3_TASK1_NEW.csv"
DS3_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/DS3_TASK2_NEW.csv"

csv_data_dict = {
    "Metric": "",
    "Model": "",
    "K": "",
    "Target": "",
    "Task 1": "",
    "Task 1 (Random Baseline)": "",
    "Task1 - Task1 baseline": "",
    "Task 2": "",
    "Task 2 (Random Baseline)": "",
    "Task2 - Task2 random": "",
    "diff_Task2-diff_Task1": "",
    "ABS(diff_Task2-diff_Task1)": "",
    "Task 2 is easy than Task1 = IF (ABS(diff_Task2-diff_Task1) < e (1e-5), \"APPROX EQUAL\", IF (diff_Task2-diff_Task1 > 0, 1, 0)": ""
}

csv_data_individual_dict = {
    "Metric": "",
    "Model": "",
    "K": "",
    "Target": "",
    "Task1_hyperparam": "",
    "Task 1": "",
    "Task 1 (Random Baseline)": "",
    "Task1 - Task1 baseline": "",
    "Task2_hyperparam": "",
    "Task 2": "",
    "Task 2 (Random Baseline)": "",
    "Task2 - Task2 random": "",
    "diff_Task2-diff_Task1": "",
    "ABS(diff_Task2-diff_Task1)": "",
    "Task 2 is easy than Task1 = IF (ABS(diff_Task2-diff_Task1) < e (1e-5), \"APPROX EQUAL\", IF (diff_Task2-diff_Task1 > 0, 1, 0)": ""
}

def getBestBaselineModelTask1(target):
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

        baseline_precision_macro = 0.2000
        baseline_recall_macro = 0.2000
        baseline_f1_macro = 0.2000
        
        class_1_weight = 0.07
        class_2_weight = 0.21
        class_3_weight = 0.27
        class_4_weight = 0.27
        class_5_weight = 0.18

        baseline_precision_weighted = 0.2261
        baseline_recall_weighted = 0.2261
        baseline_f1_weighted = 0.2261

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }
        
    return baseline_info

def DS3_getBestBaselineModel_epsilon_0(target):
    ## Config 7 ## Epsilon = 0, OverSampling = True FS=Kendalls
    ## Best baseline model = RandomGuesser based on distribution ##
    if target == "readability_level":
        baseline_0_precision = 0.493
        baseline_0_recall = 0.493
        baseline_0_f1 = 0.493

        baseline_1_precision = 0.493
        baseline_1_recall = 0.493
        baseline_1_f1 = 0.493

        baseline_2_precision = 0.014
        baseline_2_recall = 0.014
        baseline_2_f1 = 0.014

        class_0_weight = 0.493
        class_1_weight = 0.493
        class_2_weight = 0.014

        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro = 0.333

        baseline_precision_weighted = 0.487
        baseline_recall_weighted = 0.487
        baseline_f1_weighted = 0.487

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    
    return baseline_info

def DS3_getBestBaselineModel_epsilon_dynamic(target):
    ## Config 3 ## Epsilon = dynamic, OverSampling = True FS=Kendalls
    ## Best baseline model = RandomGuesser based on distribution ##
    if target == "readability_level":
        baseline_0_precision = 0.494
        baseline_0_recall = 0.494
        baseline_0_f1 = 0.494

        baseline_1_precision = 0.489
        baseline_1_recall = 0.489
        baseline_1_f1 = 0.489

        baseline_2_precision = 0.017
        baseline_2_recall = 0.017
        baseline_2_f1 = 0.017

        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro = 0.333

        baseline_precision_weighted = 0.484
        baseline_recall_weighted = 0.484
        baseline_f1_weighted = 0.484


        class_0_weight = 0.49
        class_1_weight = 0.49
        class_2_weight = 0.02

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    
    return baseline_info


def getBestBaselineModelTask1_DS6(target):
    ## best baseline = lazy0
    if target == "ABU":
        baseline_precision_macro = np.nan
        baseline_recall_macro = 0.5
        baseline_f1_macro = 0.4521

        baseline_precision_weighted = np.nan
        baseline_recall_weighted = 0.825
        baseline_f1_weighted = 0.7459

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }
        
    ## best baseline = random(distribution)    
    elif target == "ABU50":
        baseline_precision_macro = 0.50
        baseline_recall_macro = 0.5000
        baseline_f1_macro =   0.50 

        baseline_precision_weighted = 0.5002582645
        baseline_recall_weighted = 0.5002582645
        baseline_f1_weighted = 0.5003

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)
    elif target == "BD":
        baseline_precision_macro = 0.5000
        baseline_recall_macro = 0.5000
        baseline_f1_macro =   0.5000 

        baseline_precision_weighted = 0.5005061983
        baseline_recall_weighted = 0.5005061983
        baseline_f1_weighted = 0.5005

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = lazy0
    elif target == "BD50":
        baseline_precision_macro = np.nan
        baseline_recall_macro = 0.5000
        baseline_f1_macro =   0.4437

        baseline_precision_weighted = np.nan
        baseline_recall_weighted = 0.7977272727
        baseline_f1_weighted = 0.7080

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)    
    elif target == "PBU":
        baseline_precision_macro = 0.5000
        baseline_recall_macro = 0.50000000000000000
        baseline_f1_macro =   0.5000

        baseline_precision_weighted = 0.572892562
        baseline_recall_weighted = 0.57289256198347100
        baseline_f1_weighted = 0.5729

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)
    elif target == "AU":
        baseline_precision_macro = 0.2500
        baseline_recall_macro = 0.2500
        baseline_f1_macro =   0.2500

        baseline_precision_weighted = 0.2766838843
        baseline_recall_weighted = 0.2766838843
        baseline_f1_weighted = 0.2767

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    return baseline_info    

def getBestBaselineModelTask2_DS6_epsilon_0(target):
    ## best baseline = random(distribution)  
    if target == "ABU":
        baseline_precision_macro = 0.33
        baseline_recall_macro = 0.33
        baseline_f1_macro = 0.33

        baseline_precision_weighted = 0.36
        baseline_recall_weighted = 0.36
        baseline_f1_weighted = 0.36
        
        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)    
    elif target == "ABU50":
        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro =   0.333

        baseline_precision_weighted = 0.41
        baseline_recall_weighted = 0.41
        baseline_f1_weighted = 0.41

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)
    elif target == "BD":
        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro =   0.333

        baseline_precision_weighted = 0.382
        baseline_recall_weighted = 0.382
        baseline_f1_weighted = 0.382

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = lazy0
    elif target == "BD50":
        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro =   0.333

        baseline_precision_weighted = 0.373
        baseline_recall_weighted = 0.373
        baseline_f1_weighted = 0.373

        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)    
    elif target == "PBU":
        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro =   0.333

        baseline_precision_weighted = 0.396
        baseline_recall_weighted = 0.396
        baseline_f1_weighted = 0.396


        baseline_info = {
            "precision_macro": baseline_precision_macro,
            "recall_macro": baseline_recall_macro,
            "f1_macro": baseline_f1_macro,
            "precision_weighted": baseline_precision_weighted,
            "recall_weighted": baseline_recall_weighted,
            "f1_weighted": baseline_f1_weighted
        }

    ## best baseline = random(distribution)
    elif target == "AU":
        baseline_precision_macro = 0.333
        baseline_recall_macro = 0.333
        baseline_f1_macro =   0.333

        baseline_precision_weighted = 0.440
        baseline_recall_weighted = 0.440
        baseline_f1_weighted = 0.440

        baseline_info = {
                "precision_macro": baseline_precision_macro,
                "recall_macro": baseline_recall_macro,
                "f1_macro": baseline_f1_macro,
                "precision_weighted": baseline_precision_weighted,
                "recall_weighted": baseline_recall_weighted,
                "f1_weighted": baseline_f1_weighted
        }

    return baseline_info     


def dict_to_csv(output_file_path, dict_data):
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dict_data.keys())
        writer.writerow(dict_data)

## write header
with open("ml_model/src/ml-experiments/TASK2-RelativeComprehensibility/results/analysis_epsilon_0_DS6.csv", "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_data_dict.keys())
    writer.writeheader()

with open("ml_model/src/ml-experiments/TASK2-RelativeComprehensibility/results/analysis_epsilon_0_Individual_DS6.csv", "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_data_individual_dict.keys())
    writer.writeheader()


metrics = ["precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]
models = ["bayes_network",  "knn_classifier", "logisticregression", "mlp_classifier",  "randomForest_classifier", "svc"]
K = [10,20,30,40,50,60,70,80,90,100]
ds6_targets = ["AU", "ABU", "PBU", "BD", "ABU50", "BD50"]
ds3_targets = ["readability_level"]

def main():

    ## filter the DS3_TASK1 based on iteration="Overall"
     

    # ## sort the parameters in the file names of the images in the confusion matrix folder ##
    # for root, dirs, files in  os.walk("ml_model/src/ml-experiments/TASK2-RelativeComprehensibility/results/confusion_matrix/"):
    #     for file in files:
    #         if file.endswith('.png'):
    #             hyperparameters = file.split("frozenset")[1].split(".png")[0]
    #             ## sort the hyperparameters and convert to string ##
    #             hyperparameters = str(sorted(eval(hyperparameters)))
    #             ## replace the [ and ] with { and } respectively
    #             hyperparameters = "{" + hyperparameters[1:-1] + "}"
    #             new_file_name = file.split("frozenset")[0] + "frozenset(" +hyperparameters + ").png"
    #             os.rename(os.path.join(root, file), os.path.join(root, new_file_name))

    # ## sort the parameters in the file names of the images in the confusion matrix folder ##
    # for root, dirs, files in  os.walk("ml_model/src/ml-experiments/TASK1-AbsoluteTargets/Classifical-ML-Models/results/confusion_matrix/"):
    #     for file in files:
    #         if file.endswith('.png'):
    #             hyperparameters = file.split("frozenset")[1].split(".png")[0]
    #             ## sort the hyperparameters and convert to string ##
    #             hyperparameters = str(sorted(eval(hyperparameters)))
    #             ## replace the [ and ] with { and } respectively
    #             hyperparameters = "{" + hyperparameters[1:-1] + "}"
    #             new_file_name = file.split("frozenset")[0] + "frozenset(" +hyperparameters + ").png"
    #             os.rename(os.path.join(root, file), os.path.join(root, new_file_name))

                
    DS3_TASK1 = pd.read_csv(DS3_TASK1_PATH)
    DS3_TASK2 = pd.read_csv(DS3_TASK2_PATH)

    for x in DS3_TASK1["F1_weighted > baseline"].unique():
        iteration = "Overall"
        filtered_df = DS3_TASK1.query("`F1_weighted > baseline` == @x")
        for target in ds3_targets:
            filtered_df_target_model = filtered_df.query("target == @target & iteration == @iteration").sort_values(by="model")
            
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize =(10, 5))
            ## subplot for each model ##
            for model in filtered_df_target_model["model"].unique():
                filtered_df_target_model_model = filtered_df_target_model.query("model == @model")
                plt.boxplot(filtered_df_target_model_model["RI = (F1_weighted - baseline)/baseline"], positions=[model], showmeans=True, meanline=True)
                ## show mean values in the box plot ##
                y = filtered_df_target_model_model["RI = (F1_weighted - baseline)/baseline"]
                x = np.random.normal(i+1, 0.04, size=len(y))
                plt.plot(x, y, 'r.', alpha=0.5)


            filtered_df_target_model.boxplot(column=["RI = (F1_weighted - baseline)/baseline"], by=["model"], ax=fig.gca())


            plt.title("DS3-Task1 (F1_weighted > baseline = " + str(x) + ")")
            plt.ylabel("RI=(F1_weighted - baseline)/baseline")
            plt.xlabel("Model")
            text = ""
            for model in filtered_df_target_model["model"].unique():
                plt.suptitle( "Number of data points in each box plot" , fontsize=10)
                n = filtered_df_target_model.query("model == @model").shape[0]
                text+=str(n) + "  "
            plt.suptitle(text, fontsize=10)
            plt.savefig(ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/box_plot_DS3_TASK1_" + target + "_RI_" + str(x) + ".png")
            plt.close()



    ## draw box plots for the RI = (F1_weighted - baseline)/baseline of the DS3_TASK1. order by F1_weighted > baseline and model ##
    ## here there should be two box plots side by side per model. One for F1_weighted > baseline = 0 and other for F1_weighted > baseline = 0
    ## x-axis = model, y-axis = RI = (F1_weighted - baseline)/baseline
    ## title = DS3-Task1 (F1_weighted > baseline = 0) and DS3-Task1 (F1_weighted > baseline = 1)
    ## save the box plots in the results folder
    # for x in DS3_TASK1["F1_weighted > baseline"].unique():
    #     iteration = "Overall"
    #     filtered_df = DS3_TASK1.query("`F1_weighted > baseline` == @x")
    #     for target in ds3_targets:
    #         filtered_df_target_model = filtered_df.query("target == @target & iteration == @iteration").sort_values(by="model")
            
    #         import matplotlib.pyplot as plt
    #         fig = plt.figure(figsize =(10, 5))
    #         filtered_df_target_model.boxplot(column=["RI = (F1_weighted - baseline)/baseline"], by=["model"], ax=fig.gca())
    #         ## show mean values in the box plot ##
    #         for i, model in enumerate(filtered_df_target_model["model"].unique()):
    #             y = filtered_df_target_model.query("model == @model")["RI = (F1_weighted - baseline)/baseline"]
    #             x = np.random.normal(i+1, 0.04, size=len(y))
    #             plt.plot(x, y, 'r.', alpha=0.5)

    #         plt.title("DS3-Task1 (F1_weighted > baseline = " + str(x) + ")")
    #         plt.ylabel("RI=(F1_weighted - baseline)/baseline")
    #         plt.xlabel("Model")
    #         text = ""
    #         for model in filtered_df_target_model["model"].unique():
    #             plt.suptitle( "Number of data points in each box plot" , fontsize=10)
    #             n = filtered_df_target_model.query("model == @model").shape[0]
    #             text+=str(n) + "  "
    #         plt.suptitle(text, fontsize=10)
    #         plt.savefig(ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/box_plot_DS3_TASK1_" + target + "_RI_" + str(x) + ".png")
    #         plt.close()
    












    # ## sort the hyperparameters and convert to string and replace to the dataframe
    # DS3_TASK1["hyperparameters"] = DS3_TASK1["hyperparameters"].apply(lambda x: str(sorted(eval(x))))
    # DS3_TASK2["hyperparameters_c"] = DS3_TASK2["hyperparameters_c"].apply(lambda x: str(sorted(eval(x))))

    # DS3_TASK1.to_csv(ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/DS6_TASK1_HYPER_SORTED.csv", index=False)
    # DS3_TASK2.to_csv(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/DS6_TASK2_HYPER_SORTED.csv", index=False)
    
    # iteration_TASK2 = "overall"
    # iteration_TASK1 = "Overall"
    # dynamic_epsilon = False

    
    
    # for metric in metrics:
    #     for model in models:
    #         for k in K:
    #             for target in targets:
                    
    #                 baseline_TASK2 = getBestBaselineModelTask1_DS6(target)
    #                 baseline_TASK1 = getBestBaselineModelTask2_DS6_epsilon_0(target)
                    
    #                 task1_row = DS3_TASK1.query("model == @model & K == @k & target == @target & iteration == @iteration_TASK1")
    #                 task2_row = DS3_TASK2.query("model == @model & K == @k & target == @target & iteration == @iteration_TASK2 & dynamic_epsilon == @dynamic_epsilon")
                    
    #                 if not task1_row.empty and not task2_row.empty:
                        
    #                     task1_metric_value = np.nanmean(task1_row[metric].values)
    #                     task2_metric_value = np.nanmean(task2_row[metric].values)   
                        
    #                     csv_data_dict["Metric"] = metric
    #                     csv_data_dict["Model"] = model
    #                     csv_data_dict["K"] = k
    #                     csv_data_dict["Target"] = target
    #                     csv_data_dict["Task 1"] = task1_metric_value
    #                     csv_data_dict["Task 1 (Random Baseline)"] = baseline_TASK1[metric]
    #                     csv_data_dict["Task1 - Task1 baseline"] = task1_metric_value - baseline_TASK1[metric]
    #                     csv_data_dict["Task 2"] = task2_metric_value
    #                     csv_data_dict["Task 2 (Random Baseline)"] = baseline_TASK2[metric]
    #                     csv_data_dict["Task2 - Task2 random"] = task2_metric_value - baseline_TASK2[metric]
    #                     csv_data_dict["diff_Task2-diff_Task1"] = (task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])
    #                     csv_data_dict["ABS(diff_Task2-diff_Task1)"] = abs((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric]))
    #                     csv_data_dict["Task 2 is easy than Task1 = IF (ABS(diff_Task2-diff_Task1) < e (1e-5), \"APPROX EQUAL\", IF (diff_Task2-diff_Task1 > 0, 1, 0)"] = "APPROX EQUAL" if abs((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])) < 1e-5 else 1 if ((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])) > 0 else 0

    #                     dict_to_csv(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/analysis_epsilon_0_DS6.csv", csv_data_dict)


    #                     for index_task1,model_param_task1 in task1_row.iterrows():
    #                         for index_task2, model_param_task2 in task2_row.iterrows():
    #                             task1_metric_value = model_param_task1[metric]
    #                             task2_metric_value = model_param_task2[metric]

    #                             csv_data_individual_dict["Metric"] = metric
    #                             csv_data_individual_dict["Model"] = model
    #                             csv_data_individual_dict["K"] = k
    #                             csv_data_individual_dict["Target"] = target

    #                             csv_data_individual_dict["Task1_hyperparam"] = model_param_task1["hyperparameters"]
    #                             csv_data_individual_dict["Task 1"] = task1_metric_value
    #                             csv_data_individual_dict["Task 1 (Random Baseline)"] = baseline_TASK1[metric]
    #                             csv_data_individual_dict["Task1 - Task1 baseline"] = task1_metric_value - baseline_TASK1[metric]

    #                             csv_data_individual_dict["Task2_hyperparam"] = model_param_task2["hyperparameters_c"]
    #                             csv_data_individual_dict["Task 2"] = task2_metric_value
    #                             csv_data_individual_dict["Task 2 (Random Baseline)"] = baseline_TASK2[metric]
    #                             csv_data_individual_dict["Task2 - Task2 random"] = task2_metric_value - baseline_TASK2[metric]

    #                             csv_data_individual_dict["diff_Task2-diff_Task1"] = (task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])
    #                             csv_data_individual_dict["ABS(diff_Task2-diff_Task1)"] = abs((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric]))

    #                             csv_data_individual_dict["Task 2 is easy than Task1 = IF (ABS(diff_Task2-diff_Task1) < e (1e-5), \"APPROX EQUAL\", IF (diff_Task2-diff_Task1 > 0, 1, 0)"] = "APPROX EQUAL" if abs((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])) < 1e-5 else 1 if ((task2_metric_value - baseline_TASK2[metric]) - (task1_metric_value - baseline_TASK1[metric])) > 0 else 0

    #                             dict_to_csv(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/analysis_epsilon_0_Individual_DS6.csv", csv_data_individual_dict)



if __name__ == "__main__":
    main()
