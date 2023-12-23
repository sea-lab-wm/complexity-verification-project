import pandas as pd
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ptitprince as pt

from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta

ROOT_PATH = "/verification_project/"


feature_sets = ["set1", "set2", "set3", "set4"]
warning_features = ["warnings_checker_framework", "warnings_infer", "warnings_openjml", "warnings_typestate_checker", "warning_sum"]
targets = ["TNPU", "TAU", "AU"]
models = ["svm", "mlp", "linear_regression", "knn", "random_forest"]

metrics = {'mse': ['mse_c', 'mse_cw'], 
           'mae': ['mae_c', 'mae_cw'], 
           }

overall_result_header_across_models = {
    'warning_feature':'', 
    'feature_set':'',
    '#best_hyperparameters':'',
    'target':'',
    'metric':'',
    'code(median)':'', 'code+warning(median)':'',
    'code(mean)':'', 'code+warning(mean)':'',
    'diff(median)':'',
    'diff(mean)':'',
    'wilcoxon_test(p-value)':'',
    'cliffs_delta':'',
    'cliffs_delta_result':'',
    'num_of_pairs':''
}

overall_result_header_model_wise = {
    'warning_feature':'', 
    'feature_set':'',
    '#best_hyperparameters':'',  
    'model':'', 
    'target':'',
    'metric':'',
    'code(median)':'', 'code+warning(median)':'',
    'code(mean)':'', 'code+warning(mean)':'',
    'diff(median)':'',
    'diff(mean)':'',
    'wilcoxon_test(p-value)':'',
    'cliffs_delta':'',
    'cliffs_delta_result':'',
    'num_of_pairs':''
}

INPUT_FILE_NAME='Final_regression.csv'
OUTPUT_FILE_ACROSS_MODELS='Overall_results_across_model.csv'
OUTPUT_FILE_PER_MODEL = 'Overall_results_per_model.csv'

# write header
with open(ROOT_PATH + 'Results/regression/' + OUTPUT_FILE_ACROSS_MODELS, "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header_across_models.keys())
    writer.writeheader()

with open(ROOT_PATH + 'Results/regression/' + OUTPUT_FILE_PER_MODEL, "w+") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header_model_wise.keys())
    writer.writeheader()    
    
df = pd.read_csv(ROOT_PATH + 'Results/regression/'+ INPUT_FILE_NAME)

for feature_set in feature_sets:
    for target in targets:
        for warning_feature in warning_features:
      
            #####################################
            # Across model result distribution ##
            #####################################
            for metric in metrics.keys():
                ## filter overall results according feature set, warning feature, target
                    filtered_df = df.query("feature_set == '" + feature_set + "'" + " and iteration == 'overall' and target == '" + target + "'" + " and warning_feature =='" + warning_feature + "'")
                 
                    ## Filtered for code metrics not non values ##
                    filtered_df_c = filtered_df.dropna(subset=[metrics[metric][0]])
                    ## Filtered for code+warning metrics not non values ##
                    filtered_df_cw = filtered_df.dropna(subset=[metrics[metric][1]])
                    ## Filtered for difference betweetn df_weighted metrics not non values ##
                    filtered_df_diff = filtered_df.dropna(subset=["diff_" + metric])


                    ## get the code and code+warning metrics
                    c_metric = metrics[metric][0] ## eg. mae_c
                    cw_metric = metrics[metric][1] ## eg. mae_cw
                    diff_metric = "diff_" + metric ## eg. diff_mae

                    ## number of data points
                    number_of_best_hyperparameters_code = filtered_df_c.shape[0]
                    number_of_best_hyperparameters_code_code_warning = filtered_df_cw.shape[0]
                    number_of_data_points_in_both_c_cw_non_nan = filtered_df_diff[diff_metric].shape[0]
                    

                    ## compute the mean of each metric
                    metric_c_mean = filtered_df_c[c_metric].mean()
                    metric_cw_mean = filtered_df_cw[cw_metric].mean()
                    diff_metric_mean = metric_cw_mean - metric_c_mean

                    ## compute the median of each metric
                    metric_c_median = filtered_df_c[c_metric].median()
                    metric_cw_median = filtered_df_cw[cw_metric].median()
                    diff_metric_median = metric_cw_median - metric_c_median

                    ## since wilcoxon test omits pairs with at least one nan value, we need to drop nan values for both the columns
                    specific_model_data_wilcoxon = filtered_df.dropna(subset=[c_metric, cw_metric])
                    
                    x = specific_model_data_wilcoxon[c_metric] ## code data
                    y = specific_model_data_wilcoxon[cw_metric] ## code+warnings data

                    ## 1. WILCOXON TEST ##
                    ## x=code and y=code+warnings
                    ## H0: x - y is symmetric about zero. Means both x, y comes from the same distribution.
                    ## In other words The error (based on a metric) of the model using code 
                    ## features is equal to the error using code+warnings
                    ## if p-value <= 0.05, we reject the H0 this is doing alternative='greater'
                    _, wilcoxon_p = wilcoxon(x, y, alternative='greater', zero_method='zsplit', nan_policy='omit')

                    ## 2. CLIFF'S DELTA ##
                    ## delta_cliffs can be a continous value in between [-1,1]
                    ## x=code and y=code+warnings
                    ## delta_cliffs = 0 => no difference between x and y
                    ## 0 > delta_cliffs >= 1 => x group tends to have higher values than y group
                    ## 1 =< delta_cliffs < 0 => y group tends to have higher values than x group

                    ## |delta_cliffs| < 0.147 => res_cliffs incidicates negligible difference
                    ## 0.147 <= |delta_cliffs| < 0.33 => res_cliffs incidicates small difference
                    ## 0.33 <= |delta_cliffs| < 0.474 => res_cliffs incidicates medium difference
                    ## 0.474 <= |delta_cliffs| => res_cliffs incidicates large difference
                    ## source: https://www.researchgate.net/figure/Interpretation-of-Cliffs-delta-value_tbl2_335425733
                    if x.shape[0] == 0 or y.shape[0] == 0:
                        delta_cliffs = 0
                        res_cliffs = 'no data points'
                    else:    
                        delta_cliffs, res_cliffs = cliffs_delta(x, y)

                    ###############
                    ## BOX PLOTS ##
                    ###############

                    ## Individual box plots for code and code+warning metrics ##
                    ############################################################
                    plt.figure(figsize=(15, 8))
                    
                    plt.boxplot([filtered_df_c[c_metric], filtered_df_cw[cw_metric]], labels=[c_metric, cw_metric])
                    plt.title(feature_set + " | " + warning_feature + " | " + '\n' + " | #data points " + c_metric + ":" + str(number_of_best_hyperparameters_code) + " | #data points " + cw_metric + ":" + str(number_of_best_hyperparameters_code_code_warning) + "\n" + 
                            "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + "\n" +  
                            " | Cliff's result: " + str(res_cliffs) + "|" + "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(metric + " Improvement (Less is better)")
                    plt.grid(True)
                    plt.scatter(1, metric_c_mean, marker='o', color='red', s=10)
                    plt.scatter(2, metric_cw_mean, marker='o', color='red', s=10)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target  )
                    plt.savefig(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target +  '/' + metric + '-improvement.png')
                    plt.clf()
                    plt.close()

                    ## Individual raincloud plots for code and code+warning metrics ##
                    ##################################################################
                    plt.figure(figsize=(15, 8))
                    ## reshape the data frame to plot raincloud plot
                    dff=pd.melt(filtered_df, value_vars=[c_metric, cw_metric], var_name='metric', value_name='score')
                    pt.RainCloud(data=dff, x = "metric", y = "score", jitter=0, palette=["blue", "orange"],
                                    box_showmeans=True, box_meanprops=dict(marker='o', markerfacecolor='red', markersize=10),
                                    box_medianprops = dict(color = "red", linewidth = 1.5))
                    plt.title(feature_set + " | " + warning_feature + " | " + target + " | #data points " + c_metric + ":" + str(number_of_best_hyperparameters_code) + " | #data points " + cw_metric + ":" + str(number_of_best_hyperparameters_code_code_warning) + "\n" +
                            "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + "\n" +  
                            " | Cliff's result: " + str(res_cliffs) + "|" + "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(metric + " Improvement (Less is better)")
                    plt.grid(True)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target  )

                    plt.savefig(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target + '/' + metric + '-improvement.png')
                    plt.clf()
                    plt.close()


                    ## Box plot for diff_{metric} ##
                    ################################
                    plt.figure(figsize=(15, 8))

                    plt.boxplot([filtered_df_diff[diff_metric]], labels=[diff_metric])
                    plt.title(feature_set + " | " + warning_feature + " | " + target + " | #data points "+ diff_metric + ":" + str(number_of_data_points_in_both_c_cw_non_nan))
                    plt.ylabel("Diff " + diff_metric + " Improvement")
                    plt.grid(True)
                    plt.scatter(1, filtered_df_diff[diff_metric].mean(), marker='o', color='red', s=10)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" +  feature_set + "-" + warning_feature + "-" + target):
                        os.makedirs(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target )
                    plt.savefig(ROOT_PATH + 'Results/regression/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target + '/'+ diff_metric + '-improvement.png')
                    plt.clf()
                    plt.close()

                    ## raincloud plots for diff_{metric} ##
                    ####################################### 
                    plt.figure(figsize=(15, 8))

                    ## reshape the data frame to plot raincloud plot
                    dff=pd.melt(filtered_df, value_vars=[diff_metric], var_name='metric', value_name='score')

                    pt.RainCloud(data=dff, x = "metric", y = "score", jitter=0, palette=["blue"],
                                    box_showmeans=True, box_meanprops=dict(marker='o', markerfacecolor='red', markersize=10),
                                    box_medianprops = dict(color = "red", linewidth = 1.5))

                    plt.title(feature_set + " | " + warning_feature + " | " + target + '\n' + " | #data points "+ diff_metric + ":" + str(number_of_data_points_in_both_c_cw_non_nan))
                    plt.ylabel("Diff " + diff_metric + " Improvement")
                    plt.grid(True)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target)
       
                    plt.savefig(ROOT_PATH + 'Results/regression/raincloud/across-model-result-distribution/' + metric + "-" + feature_set + "-" + warning_feature + "-" + target + '/' + diff_metric + '-improvement.png')
                    plt.clf()
                    plt.close()

                      

                    ## Result write to csv file ##
                    overall_result_header_across_models['warning_feature'] = warning_feature
                    overall_result_header_across_models['feature_set'] = feature_set
                    overall_result_header_across_models['#best_hyperparameters'] = "c" + ' (' + str(number_of_best_hyperparameters_code) + ') ' + "cw" + ' (' + str(number_of_best_hyperparameters_code_code_warning) + ')'
                    
                    overall_result_header_across_models['target'] = target
                    overall_result_header_across_models['metric'] = metric
                    overall_result_header_across_models['code(median)'] = metric_c_median
                    overall_result_header_across_models['code+warning(median)'] = metric_cw_median
                    overall_result_header_across_models['code(mean)'] = metric_c_mean
                    overall_result_header_across_models['code+warning(mean)'] = metric_cw_mean
                    overall_result_header_across_models['diff(median)'] = diff_metric_median
                    overall_result_header_across_models['diff(mean)'] = diff_metric_mean
                    overall_result_header_across_models['wilcoxon_test(p-value)'] = wilcoxon_p
                    overall_result_header_across_models['cliffs_delta'] = delta_cliffs
                    overall_result_header_across_models['cliffs_delta_result'] = res_cliffs
                    overall_result_header_across_models['num_of_pairs'] = x.shape[0]

                    with open(ROOT_PATH + 'Results/regression/' + OUTPUT_FILE_ACROSS_MODELS, "a") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=overall_result_header_across_models.keys())
                        writer.writerow(overall_result_header_across_models)

            ####################################
            ## Model wise result distribution ##
            ####################################
            for model in models:

                ## filter overall results according feature set, warning feature, target, and model
                filtered_df = df.query("feature_set == '" + feature_set + "'" + " and iteration == 'overall' and target == '" + target + "'" + " and warning_feature =='" + warning_feature + "'" + "and model == '" + model + "'")

                for metric in metrics.keys():
                    ## get the code and code+warning metrics
                    c_metric = metrics[metric][0]
                    cw_metric = metrics[metric][1]
                    diff_metric = "diff_" + metric

                    ## filter the data frame for code and code+warning metrics
                    filtered_df_c = filtered_df.dropna(subset=[c_metric])
                    filtered_df_cw = filtered_df.dropna(subset=[cw_metric])
                    filtered_df_diff = filtered_df.dropna(subset=[diff_metric])

                    ## number of data points
                    number_of_best_hyperparameters_code = filtered_df_c.shape[0]
                    number_of_best_hyperparameters_code_code_warning = filtered_df_cw.shape[0]
                    number_of_data_points_in_both_c_cw_non_nan = filtered_df_diff[diff_metric].shape[0]

                    ## compute the mean of each metric
                    metric_c_mean = filtered_df_c[c_metric].mean()
                    metric_cw_mean = filtered_df_cw[cw_metric].mean()
                    diff_metric_mean = metric_cw_mean - metric_c_mean

                    ## compute the median of each metric
                    metric_c_median = filtered_df_c[c_metric].median()
                    metric_cw_median = filtered_df_cw[cw_metric].median()
                    diff_metric_median = metric_cw_median - metric_c_median

                    ## since wilcoxon test omits pairs with at least one nan value, we need to drop nan values for both the columns
                    specific_model_data_wilcoxon = filtered_df.dropna(subset=[c_metric, cw_metric])

                    x = specific_model_data_wilcoxon[c_metric] ## code data
                    y = specific_model_data_wilcoxon[cw_metric] ## code+warnings data

                    ## 1. WILCOXON TEST ##
                    ## x=code and y=code+warnings
                    ## H0: x - y is symmetric about zero. Means both x, y comes from the same distribution.
                    ## In other words The error (based on a metric) of the model using code 
                    ## features is equal to the error using code+warnings
                    ## if p-value <= 0.05, we reject the H0 this is doing alternative='greater'
                    _, wilcoxon_p = wilcoxon(x, y, alternative='greater', zero_method='zsplit', nan_policy='omit')

                    ## 2. CLIFF'S DELTA ##
                    ## delta_cliffs can be a continous value in between [-1,1]
                    ## x=code and y=code+warnings
                    ## delta_cliffs = 0 => no difference between x and y
                    ## 0 > delta_cliffs >= 1 => x group tends to have higher values than y group
                    ## -1 =< delta_cliffs < 0 => y group tends to have higher values than x group

                    ## |delta_cliffs| < 0.147 => res_cliffs incidicates negligible difference
                    ## 0.147 <= |delta_cliffs| < 0.33 => res_cliffs incidicates small difference
                    ## 0.33 <= |delta_cliffs| < 0.474 => res_cliffs incidicates medium difference
                    ## 0.474 <= |delta_cliffs| => res_cliffs incidicates large difference
                    ## source: https://www.researchgate.net/figure/Interpretation-of-Cliffs-delta-value_tbl2_335425733
                    if x.shape[0] == 0 or y.shape[0] == 0:
                        delta_cliffs = 0
                        res_cliffs = 'no data points'
                    else:    
                        delta_cliffs, res_cliffs = cliffs_delta(x, y)


                    ## Individual box plots for code and code+warning metrics ##
                    ############################################################
                    plt.figure(figsize=(15, 8))
                    plt.boxplot([filtered_df_c[c_metric], filtered_df_cw[cw_metric]], labels=[c_metric, cw_metric])
                    plt.title(feature_set + " | " + model + " | " + warning_feature + " | " + target + '\n' + " | #data points " + c_metric + ": " + str(number_of_best_hyperparameters_code) + " | #data points " + cw_metric + " : " + str(number_of_best_hyperparameters_code_code_warning) + "\n"
                                  + "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + '\n' + "|" + "Clinff's result: " + str(res_cliffs) + "|" +
                                  "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(metric +" Improvement (Less is better)")
                    plt.grid(True)
                    plt.scatter(1, metric_c_mean, marker='o', color='red', s=100)
                    plt.scatter(2, metric_cw_mean, marker='o', color='red', s=100)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target )

                    plt.savefig(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target + '/' + metric + '.png')
                    plt.clf()
                    plt.close()


                        
                    ## Individual raincloud plots for code and code+warning metrics ##
                    ##################################################################
                    plt.figure(figsize=(15, 8))

                    ## reshape the data frame to plot raincloud plot
                    dff=pd.melt(filtered_df, value_vars=[c_metric, cw_metric], var_name='metric', value_name='score')
                        
                    pt.RainCloud(data=dff, x = "metric", y = "score", jitter=0, palette=["blue", "orange"], 
                        box_showmeans=True, box_meanprops=dict(marker='o', markerfacecolor='red', markersize=10),
                        box_medianprops = dict(color = "red", linewidth = 1.5))
                        
                    plt.title(feature_set + " | " + model + " | " + warning_feature + " | " + target + " | #data points " + c_metric + ": " + str(number_of_best_hyperparameters_code) + " | #data points " + cw_metric + " : " + str(number_of_best_hyperparameters_code_code_warning) + "\n"
                        + "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + '\n' + "|" + "Clinff's result: " + str(res_cliffs) + "|" +
                        "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(metric +" Improvement (Less is better)")
                    plt.grid(True)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')
                        
                    if not os.path.exists(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target )    

                    plt.savefig(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target + '/' + metric + '.png')
                    plt.clf()
                    plt.close()

                    ## Box plot for diff_{metric} ##
                    ################################
                    plt.figure(figsize=(15, 8))
                    plt.boxplot([filtered_df_diff[diff_metric]], labels=[diff_metric])
                    plt.title(feature_set + " | " + model + " | " + warning_feature + " | " + target + '\n' +" | #data points " + diff_metric + ": " + str(number_of_data_points_in_both_c_cw_non_nan) + "\n"
                        + "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + "|" +
                        "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(diff_metric + " Improvement")
                    plt.grid(True)
                    plt.scatter(1, filtered_df_diff[diff_metric].mean(), marker='o', color='red', s=10)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" +target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target )

                    plt.savefig(ROOT_PATH + 'Results/regression/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target + '/' + diff_metric + '-improvement.png')
                    plt.clf()
                    plt.close() 

                    ## raincloud plots for diff_{metric} ##
                    #######################################
                    plt.figure(figsize=(15, 8))

                    ## reshape the data frame to plot raincloud plot
                    dff=pd.melt(filtered_df, value_vars=[diff_metric], var_name='metric', value_name='score')

                    pt.RainCloud(data=dff, x = "metric", y = "score", jitter=0, palette=["blue"], 
                                     box_showmeans=True, box_meanprops=dict(marker='o', markerfacecolor='red', markersize=10),
                                     box_medianprops = dict(color = "red", linewidth = 1.5))
                        
                    plt.title(feature_set + " | " + model + " | " + warning_feature + " | " + target + '\n' +" | #data points " + diff_metric + ": " + str(number_of_data_points_in_both_c_cw_non_nan) + "\n"
                                    + "Wilcoxon p-value: " + str(wilcoxon_p) + " | Cliff's delta: " + str(delta_cliffs) + "|" +
                                    "num of pairs: " + str(x.shape[0]))
                    plt.ylabel(diff_metric + " Improvement")
                    plt.grid(True)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')

                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target ):
                        os.makedirs(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target )

                    plt.savefig(ROOT_PATH + 'Results/regression/raincloud/model-wise-result-distribution/' + feature_set + "-" + model + "-" + warning_feature + "-" + target + '/' + diff_metric + '-improvement.png')
                    plt.clf()
                    plt.close()


                    ## Result write to csv file ##
                    overall_result_header_model_wise['warning_feature'] = warning_feature
                    overall_result_header_model_wise['feature_set'] = feature_set
                    overall_result_header_model_wise['#best_hyperparameters'] = "c" + ' (' + str(number_of_best_hyperparameters_code) + ') ' + "cw" + ' (' + str(number_of_best_hyperparameters_code_code_warning) + ')'
                    overall_result_header_model_wise['model'] = model
                    overall_result_header_model_wise['target'] = target
                    overall_result_header_model_wise['metric'] = metric
                    overall_result_header_model_wise['code(median)'] = metric_c_median
                    overall_result_header_model_wise['code+warning(median)'] = metric_cw_median
                    overall_result_header_model_wise['code(mean)'] = metric_c_mean
                    overall_result_header_model_wise['code+warning(mean)'] = metric_cw_mean
                    overall_result_header_model_wise['diff(median)'] = diff_metric_median
                    overall_result_header_model_wise['diff(mean)'] = diff_metric_mean
                    overall_result_header_model_wise['wilcoxon_test(p-value)'] = wilcoxon_p
                    overall_result_header_model_wise['cliffs_delta'] = delta_cliffs
                    overall_result_header_model_wise['cliffs_delta_result'] = res_cliffs
                    overall_result_header_model_wise['num_of_pairs'] = x.shape[0]

                    with open(ROOT_PATH + 'Results/regression/' + OUTPUT_FILE_PER_MODEL, "a") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=overall_result_header_model_wise.keys())
                        writer.writerow(overall_result_header_model_wise)   