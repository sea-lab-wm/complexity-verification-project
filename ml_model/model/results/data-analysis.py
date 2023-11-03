import pandas as pd
import csv

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"
df = pd.read_csv(ROOT_PATH + 'results/mean_median_overall_results_new.csv')


feature_sets = ["set1", "set2", "set3", "set4"]
warning_features = ["warnings_checker_framework", "warnings_infer", "warnings_openjml", "warnings_typestate_checker", "warning_sum"]
targets = ["PBU", "ABU50", "BD50"]
smote = [True, False]
metrics = ["f1", "auc"]
models = ["SVC","knn_classifier","logistic_regression","randomForest_classifier","mlp_classifier","bayes_networkt"]

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

for feature_set in feature_sets:
    for warning_feature in warning_features:
        for target in targets:
            for smote_value in smote:

                ## filter overall results according feature set, warning feature, target, smote value and metric
                filtered_df = df.query("warning_feature == '" + warning_feature + "' and feature_set == '" + feature_set + "' and target == '" + target + "' and use_smote == " + str(smote_value))
                    
                ## filtered by F1 metric
                filtered_df_f1 = filtered_df.query("metric == 'f1'")    
                ## num of models_F1_mean_improved
                n_improved_f1_mean_models = filtered_df_f1[filtered_df_f1["diff(mean)"] > 0].shape[0]
                ## num of models_F1_median_improved
                n_improved_f1_median_models = filtered_df_f1[filtered_df_f1["diff(median)"] > 0].shape[0]
                ## num of models_improved_both_F1_(median+mean)
                n_improved_f1_mean_median_models = filtered_df_f1[(filtered_df_f1["diff(median)"] > 0) & (filtered_df_f1["diff(mean)"] > 0)].shape[0]

                ## avg(mean_F1_improvement)
                avg_mean_f1_improvement = filtered_df_f1["diff(mean)"].mean()
                ## median(mean_F1_improvement)
                median_mean_f1_improvement = filtered_df_f1["diff(mean)"].median()
                ## q3(mean_F1_improvement)
                q3_mean_f1_improvement = filtered_df_f1["diff(mean)"].quantile(0.75)

                ## avg(median_F1_improvement)
                avg_median_f1_improvement = filtered_df_f1["diff(median)"].mean()
                ## median(median_F1_improvement)
                median_median_f1_improvement = filtered_df_f1["diff(median)"].median()
                ## q3(median_F1_improvement)
                q3_median_f1_improvement = filtered_df_f1["diff(median)"].quantile(0.75)

                ## filtered by AUC metric
                filtered_df_auc = filtered_df.query("metric == 'auc'")
                n_improved_auc_mean_models = filtered_df_auc[filtered_df_auc["diff(mean)"] > 0].shape[0]
                n_improved_auc_median_models = filtered_df_auc[filtered_df_auc["diff(median)"] > 0].shape[0]
                n_improved_auc_mean_median_models = filtered_df_auc[(filtered_df_auc["diff(median)"] > 0) & (filtered_df_auc["diff(mean)"] > 0)].shape[0]

                ## avg, median, q3 of mean AUC improvement
                avg_mean_auc_improvement = filtered_df_auc["diff(mean)"].mean()
                median_mean_auc_improvement = filtered_df_auc["diff(mean)"].median()
                q3_mean_auc_improvement = filtered_df_auc["diff(mean)"].quantile(0.75)

                ## avg, median, q3 of median AUC improvement
                avg_median_auc_improvement = filtered_df_auc["diff(median)"].mean()
                median_median_auc_improvement = filtered_df_auc["diff(median)"].median()
                q3_median_auc_improvement = filtered_df_auc["diff(median)"].quantile(0.75)

                ## filtered by both F1 and AUC metrics
                filtered_df_both_metric = filtered_df.query("metric == 'auc' or metric == 'f1'")

                n_total_improved_models_both_mean_median_f1_auc = 0
                n_total_improved_models_both_mean_median_f1_auc_with_statistical_significance = 0

                for model_name in models:
                    ## filtered by model name for F1 metric
                    filtered_df_both_metric = filtered_df_f1.query("model == '" + model_name + "'")
                    n_improved_f1_both_mean_median_models = filtered_df_both_metric[(filtered_df_both_metric["diff(median)"] > 0) & (filtered_df_both_metric["diff(mean)"] > 0)].shape[0]
                    n_improved_f1_both_mean_median_with_statistical_significance_models = filtered_df_both_metric[(filtered_df_both_metric["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_both_metric["diff(median)"] > 0) & (filtered_df_both_metric["diff(mean)"] > 0)].shape[0]

                    ## filtered by model name for AUC metric
                    filtered_df_both_metric = filtered_df_auc.query("model == '" + model_name + "'")
                    n_improved_auc_both_mean_median_models = filtered_df_both_metric[(filtered_df_both_metric["diff(median)"] > 0) & (filtered_df_both_metric["diff(mean)"] > 0)].shape[0]
                    n_improved_auc_both_mean_median_with_statistical_significance_models = filtered_df_both_metric[(filtered_df_both_metric["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_both_metric["diff(median)"] > 0) & (filtered_df_both_metric["diff(mean)"] > 0)].shape[0]

                    if n_improved_f1_both_mean_median_models > 0 and n_improved_auc_both_mean_median_models > 0:
                        n_total_improved_models_both_mean_median_f1_auc += (n_improved_f1_both_mean_median_models + n_improved_auc_both_mean_median_models)

                    if n_improved_f1_both_mean_median_with_statistical_significance_models > 0 and n_improved_auc_both_mean_median_with_statistical_significance_models > 0:
                        n_total_improved_models_both_mean_median_f1_auc_with_statistical_significance += (n_improved_f1_both_mean_median_with_statistical_significance_models + n_improved_auc_both_mean_median_with_statistical_significance_models)

                n_models_improved_F1_mean_with_statistical_significance = filtered_df_f1[(filtered_df_f1["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_f1["diff(mean)"] > 0)].shape[0]
                n_models_improved_F1_median_with_statistical_significance = filtered_df_f1[(filtered_df_f1["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_f1["diff(median)"] > 0)].shape[0]
                n_models_improved_F1_mean_median_with_statistical_significance = filtered_df_f1[(filtered_df_f1["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_f1["diff(median)"] > 0) & (filtered_df_f1["diff(mean)"] > 0)].shape[0]

                n_models_improved_AUC_mean_with_statistical_significance = filtered_df_auc[(filtered_df_auc["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_auc["diff(mean)"] > 0)].shape[0]
                n_models_improved_AUC_median_with_statistical_significance = filtered_df_auc[(filtered_df_auc["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_auc["diff(median)"] > 0)].shape[0]
                n_models_improved_AUC_mean_median_with_statistical_significance = filtered_df_auc[(filtered_df_auc["wilcoxon_test(p-value)"] < 0.05) & (filtered_df_auc["diff(median)"] > 0) & (filtered_df_auc["diff(mean)"] > 0)].shape[0]
                
                overall_result_header["feature_set"] = feature_set
                overall_result_header["warning_feature"] = warning_feature
                overall_result_header["target"] = target
                overall_result_header["use_smote"] = smote_value
                overall_result_header["#models_F1_mean_improved / 6"] = n_improved_f1_mean_models
                overall_result_header["#models_F1_median_improved / 6"] = n_improved_f1_median_models
                overall_result_header["#models_improved_both_F1(median+mean) / 6"] = n_improved_f1_mean_median_models
                overall_result_header["#models_AUC_mean_improved / 6"] = n_improved_auc_mean_models
                overall_result_header["#models_AUC_median_improved / 6"] = n_improved_auc_median_models
                overall_result_header["#models_improved_both_AUC(median+mean) / 6"] = n_improved_auc_mean_median_models
                overall_result_header["#models_improved_in_F1(median+mean)+AUC(median+mean) / 12"] = n_total_improved_models_both_mean_median_f1_auc
                overall_result_header["avg(mean_F1_improvement)"] = avg_mean_f1_improvement
                overall_result_header["median(mean_F1_improvement)"] = median_mean_f1_improvement
                overall_result_header["q3(mean_F1_improvement)"] = q3_mean_f1_improvement
                overall_result_header["avg(median_F1_improvement)"] = avg_median_f1_improvement
                overall_result_header["median(median_F1_improvement)"] = median_median_f1_improvement
                overall_result_header["q3(median_F1_improvement)"] = q3_median_f1_improvement
                overall_result_header["#models_improved_F1_mean_with_statistical_significance / 6"] = n_models_improved_F1_mean_with_statistical_significance
                overall_result_header["#models_improved_F1_median_with_statistical_significance / 6"] = n_models_improved_F1_median_with_statistical_significance
                overall_result_header["#models_improved_both_F1(mean+median)_with_statistical_significance / 6"] = n_models_improved_F1_mean_median_with_statistical_significance
                overall_result_header["avg(mean_AUC_improvement)"] = avg_mean_auc_improvement
                overall_result_header["median(mean_AUC_improvement)"] = median_mean_auc_improvement
                overall_result_header["q3(mean_AUC_improvement)"] = q3_mean_auc_improvement
                overall_result_header["avg(median_AUC_improvement)"] = avg_median_auc_improvement
                overall_result_header["median(median_AUC_improvement)"] = median_median_auc_improvement
                overall_result_header["q3(median_AUC_improvement)"] = q3_median_auc_improvement
                overall_result_header["#models_improved_AUC_mean_with_statistical_significance / 6"] = n_models_improved_AUC_mean_with_statistical_significance
                overall_result_header["#models_improved_AUC_median_with_statistical_significance / 6"] = n_models_improved_AUC_median_with_statistical_significance
                overall_result_header["#models_improved_both_AUC(mean+median)_with_statistical_significance / 6"] = n_models_improved_AUC_mean_median_with_statistical_significance
                overall_result_header["#models_improved_F1(mean+median)+AUC(mean+median)_with_statistical_significance / 12"] = n_total_improved_models_both_mean_median_f1_auc_with_statistical_significance

                ## write results to csv file
                with open(ROOT_PATH + 'results/Final_Results_New.csv', "a") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
                    writer.writerow(overall_result_header)
