import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"


feature_sets = ["set1", "set2", "set3", "set4"]
warning_features = ["warnings_checker_framework", "warnings_infer", "warnings_openjml", "warnings_typestate_checker", "warning_sum"]
targets = ["PBU", "ABU50", "BD50"]
smote = [True, False]
metrics = ["f1", "auc"]
models = ["SVC","knn_classifier","logistic_regression","randomForest_classifier","mlp_classifier","bayes_network"]

overall_result_header = {
    "feature_set" : '',	
    "warning_feature": '',	
    "target": '',
    "use_smote": '',
    "#models_F1_mean_improved / 6": '',
    "#models_F1_median_improved / 6": '',
    "#models_improved_both_F1(median+mean) / 6": '',
    "#models_improved_F1_mean_with_statistical_significance / 6": '',
    "#models_improved_F1_median_with_statistical_significance / 6": '',
    "#models_improved_both_F1(mean+median)_with_statistical_significance / 6": '',
    "avg(mean_F1_improvement)": '',
    "median(mean_F1_improvement)": '',
    "q3(mean_F1_improvement)": '',
    "avg(median_F1_improvement)": '',
    "median(median_F1_improvement)": '',
    "q3(median_F1_improvement)": '',
    "#models_AUC_mean_improved / 6": '',
    "#models_AUC_median_improved / 6": '',
    "#models_improved_both_AUC(median+mean) / 6": '',
    "#models_improved_in_F1(median+mean)+AUC(median+mean) / 12": '',
    "#models_improved_AUC_mean_with_statistical_significance / 6": '',
    "#models_improved_AUC_median_with_statistical_significance / 6": '',
    "#models_improved_both_AUC(mean+median)_with_statistical_significance / 6": '',
    "#models_improved_F1(mean+median)+AUC(mean+median)_with_statistical_significance / 12": '',
    "avg(mean_AUC_improvement)": '',
    "median(mean_AUC_improvement)": '',
    "q3(mean_AUC_improvement)": '',
    "avg(median_AUC_improvement)": '',
    "median(median_AUC_improvement)": '',
    "q3(median_AUC_improvement)": '',
}

## write header
with open(ROOT_PATH + 'results/Final_Results_New.csv', "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
    writer.writeheader()
    
ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"
# df = pd.read_csv(ROOT_PATH + 'results/raw_results.csv')

# ## adding a column to the dataframe including the feature set and warning feature
# with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
#     experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
#     for experiment in experiments:
#         warning_feature = experiment["features"][0]
#         feature_set = experiment["experiment_id"].split("-")[3]
#         df.loc[df["experiment"] == experiment["experiment_id"], "warning_feature"] = warning_feature
#         df.loc[df["experiment"] == experiment["experiment_id"], "feature_set"] = feature_set

# df.to_csv(ROOT_PATH + 'results/raw_results.csv', index=False)

df = pd.read_csv(ROOT_PATH + 'results/raw_results.csv')


for feature_set in feature_sets:
    for smote_value in smote:
        for target in targets:
            for warning_feature in warning_features:
                for model in models:

                    ## filter overall results according feature set, warning feature, target, smote value and metric
                    filtered_df = df.query("feature_set == '" + feature_set + "' and iteration == 'overall' and target == '" + target + "'" + " and use_smote == " + str(smote_value) + " and warning_feature =='" + warning_feature + "' and model == '" + model + "'")

                    ## Filtered for f1_c not non values ##
                    filtered_df_f1_c = filtered_df.dropna(subset=["f1_c"])
                    ## Filtered for f1_cw not non values ##
                    filtered_df_f1_cw = filtered_df.dropna(subset=["f1_cw"])

                    filtered_df_diff_f1 = filtered_df.dropna(subset=["diff_f1"])

                    ## avg F1_c improvement
                    avg_f1_c_improvement = filtered_df_f1_c["f1_c"].mean()

                    ## avg F1_cw improvement
                    avg_f1_cw_improvement = filtered_df_f1_cw["f1_cw"].mean()


                    ## diff f1 improvement
                    avg_diff_f1 = filtered_df_diff_f1["diff_f1"].mean()

                    ## BOX PLOT FOR F1 IMPROVEMENT ##
                    plt.figure(figsize=(10, 6))

                    ## number of data points
                    n1 = filtered_df_f1_c["f1_c"].shape[0]
                    n2 = filtered_df_f1_cw["f1_cw"].shape[0]

                    plt.boxplot([filtered_df_f1_c["f1_c"], filtered_df_f1_cw["f1_cw"]], labels=["F1_c", "F1_cw"])

                    plt.title(feature_set + " | " + warning_feature + " | " + target + " | SMOTE? " +  str(smote_value) + " | #data points (F1_c): " + str(n1) + " | #data points (F1_cw): " + str(n2))

                    plt.ylabel("F1 Improvement")
                    plt.grid(True)

                    plt.scatter(1, avg_f1_c_improvement, marker='o', color='red', s=10)
                    plt.scatter(2, avg_f1_cw_improvement, marker='o', color='red', s=10)

                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" + target + "-" + str(smote_value) + "-" + model):
                        os.makedirs(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" + target + "-" + str(smote_value) + "-" + model)

                    plt.savefig(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" + target + "-" + str(smote_value) + "-" + model +  '/f1-improvement.png')
                    plt.clf()
                    plt.close()


                    ## add the box plot for diff_f1
                    plt.figure(figsize=(10, 6))

                    ## number of data points
                    n3 = filtered_df_diff_f1["diff_f1"].shape[0]

                    plt.boxplot([filtered_df_diff_f1["diff_f1"]], labels=["diff_f1"])

                    plt.title(feature_set + " | " + warning_feature + " | " + target + " | SMOTE? " +  str(smote_value) + " | #data points (diff_f1): " + str(n3))

                    plt.ylabel("Diff F1 Improvement")
                    plt.grid(True)

                    plt.scatter(1, avg_diff_f1, marker='o', color='red', s=10)

                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" +target + "-" + str(smote_value) + "-" + model):
                        os.makedirs(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" + target + "-" + str(smote_value) + "-" + model)

                    plt.savefig(ROOT_PATH + 'results/all-results/model-wise-result-distribution/' + feature_set + "-" + warning_feature + "-" + target + "-" + str(smote_value) + "-" + model +  '/diff-f1-improvement.png')
                    plt.clf()
                    plt.close()    





