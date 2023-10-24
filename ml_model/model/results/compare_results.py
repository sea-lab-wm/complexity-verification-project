'''
This file is used to process the raw_results from the experiments 
and generate the overall results, boxplots ,and statistical tests.
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import csv
import json
import os

from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"
df = pd.read_csv(ROOT_PATH + 'results/raw_results.csv')


with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:

    overall_result_header = {
        'Experiment':'', 
        'Warning Feature':'', 'Feature Set':'',
        '#best hyperparameters':'', 
        'use_smote':'', 
        'Model':'', 
        'Target':'', 
        'f1_c (median)':'', 'f1_cw (median)':'', 
        'precision_c (median)':'', 'precision_cw (median)':'',
        'recall_c (median)':'', 'recall_cw (median)':'', 
        'auc_c (median)':'', 'auc_cw (median)':'',
        'f1_c (mean)':'', 'f1_cw (mean)':'',
        'precision_c (mean)':'', 'precision_cw (mean)':'',
        'recall_c (mean)':'', 'recall_cw (mean)':'',
        'auc_c (mean)':'', 'auc_cw (mean)':'',
        'diff_f1 (median)':'',
        'diff_f1 (mean)':'',
        'diff_auc (median)':'', 
        'diff_auc (mean)':'',
        'diff_precision (median)':'', 
        'diff_precision (mean)':'', 
        'diff_recall (median)':'',
        'diff_recall (mean)':''
        }
    
    overall_test_results = {'Experiment':'', 'use_SMOTE':'', 'metric':'', 'model':'', 'target':'', '#best hyperparameters': 0, 'wilcoxon_test(p-value)':'', 'cliffs_delta':'', 'cliffs_delta_result':''}

    ## write header
    with open(ROOT_PATH + 'results/mean_median_overall_results.csv', "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
        writer.writeheader()

    ## write header
    with open(ROOT_PATH + 'results/final_statistics.csv', "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=overall_test_results.keys())
        writer.writeheader()    

    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
    for experiment in experiments:

        ## experiment id
        experiment_id = experiment["experiment_id"].split("-")[0] ## eg: exp1

        use_SMOTE = experiment["use_SMOTE"]

        if use_SMOTE: 
            SMOTE_TAG='SMOTE'
        else:
            SMOTE_TAG='NO-SMOTE'

        ############################
        ## FILTER OVERALL RESULTS ##
        ############################

        ## filter overall results according to the experiment id and use_SMOTE
        overall_df = df.query("iteration == 'overall' and use_smote == " + str(use_SMOTE) + " and experiment=='" + experiment["experiment_id"] + "'")

        ## create results folder if it doesn't exist in results directory
        if not os.path.exists(ROOT_PATH + 'results/results-'+ experiment_id):
            os.makedirs(ROOT_PATH + 'results/results-'+ experiment_id)

        ## save the results
        overall_df.to_csv(ROOT_PATH + 'results/results-'+ experiment_id + '/overall_results_'+ str(SMOTE_TAG) + '.csv', index=False)

        # Group the data by target and find the rows with the maximum F1-scores for each target 
        best_code_features = overall_df.loc[overall_df.groupby('target')['f1_c'].idxmax()]
        best_code_warning_features = overall_df.loc[overall_df.groupby('target')['f1_cw'].idxmax()]
        
        # Write the results to a CSV file
        best_code_features.to_csv(ROOT_PATH + 'results/results-'+ experiment_id +'/best_code_models_'+ str(SMOTE_TAG) + '.csv', index=False)
        best_code_warning_features.to_csv(ROOT_PATH + 'results/results-'+ experiment_id + '/best_code+warning_models_'+ str(SMOTE_TAG) + '.csv', index=False)

        targets = [experiment["target"]]

        metrics = {'f1': ['f1_c','f1_cw'],'precision': ['precision_c','precision_cw'],
                'recall': ['recall_c','recall_cw'], 'auc': ['auc_c','auc_cw']}
        
        models = ['SVC', 'knn_classifier', 'logistic_regression', 'randomForest_classifier', 'mlp_classifier', 'bayes_network']

        
        for model_name in models:
            for target in targets:
                model_data = overall_df.query("model == '" + model_name + "' and target == '" + target + "'")
                
                ## number of data points
                ## we consider both hyperparameters obtained for code and code+warnings
                number_of_best_hyperparameters = model_data.shape[0]

                ## write overall results to csv
                overall_result_header['Experiment'] = experiment["experiment_id"]
                overall_result_header['Warning Feature'] = experiment["features"][0]
                overall_result_header['Feature Set'] = experiment["experiment_id"].split("-")[3] ## eg: set1
                overall_result_header['#best hyperparameters'] = number_of_best_hyperparameters
                overall_result_header['use_smote'] = use_SMOTE
                overall_result_header['Model'] = model_name
                overall_result_header['Target'] = target

                overall_test_results['Experiment'] = experiment["experiment_id"]
                overall_test_results['use_SMOTE'] = use_SMOTE
                overall_test_results['model'] = model_name
                overall_test_results['target'] = target
                overall_test_results['#best hyperparameters'] = number_of_best_hyperparameters


                for metric in metrics.keys():
                    ## compute the mean and median values for metrics for both code and code+warnings
                    c_metric = metrics[metric][0]
                    cw_metric = metrics[metric][1]

                    x = model_data[c_metric] ## code
                    y = model_data[cw_metric] ## code+warnings

                    metric_c_median = model_data[c_metric].median()
                    metric_cw_median = model_data[cw_metric].median()

                    metric_c_mean = model_data[c_metric].mean() ## since we don't have outliers we can reliably use the mean
                    metric_cw_mean = model_data[cw_metric].mean()

                    ## median values
                    print(model_data.groupby(['model','target'])[metrics[metric]].median())
                    overall_result_header[f'{c_metric} (median)'] = metric_c_median
                    overall_result_header[f'{cw_metric} (median)'] = metric_cw_median
                    
                    ## mean values
                    print(model_data.groupby(['model','target'])[metrics[metric]].mean())
                    overall_result_header[f'{c_metric} (mean)']  = metric_c_mean
                    overall_result_header[f'{cw_metric} (mean)']  = metric_cw_mean

                    overall_result_header[f'diff_{metric} (median)'] = metric_c_median - metric_cw_median ## diff = code - code+warnings
                    overall_result_header[f'diff_{metric} (mean)'] = metric_c_mean - metric_cw_mean ## diff = code - code+warnings

                    ##############
                    ## BOX PLOTS ##
                    ##############
                    ## show the mean and median values as dots for both code and code+warnings in the boxplots
                    plt.figure(figsize=(10, 6))
                    plt.boxplot([model_data[c_metric], model_data[cw_metric]], labels=[c_metric, cw_metric])
                    plt.title(f'Target: {target} | Model: {model_name} | #best hyperparameters (data points): {number_of_best_hyperparameters}')
                    plt.ylabel(metric)
                    plt.grid(True)
                    plt.scatter(1, metric_c_mean, marker='o', color='red', s=10)
                    plt.scatter(2, metric_cw_mean, marker='o', color='red', s=10)
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mean', markerfacecolor='r', markersize=10)]
                    plt.legend(handles=legend_elements, loc='upper right')
                    ## create final-boxplots folder if it doesn't exist in results directory
                    if not os.path.exists(ROOT_PATH + 'results/results-' + experiment_id + '/final-boxplots'):
                        os.makedirs(ROOT_PATH + 'results/results-' + experiment_id + '/final-boxplots')
                    plt.savefig(ROOT_PATH + 'results/results-' + experiment_id + '/final-boxplots/' + target + '_' + model_name + '_' + metric +'_boxplot_'+ str(SMOTE_TAG) + '.png')
                    plt.close() 


                    #######################
                    ## STATISTICAL TESTS ##
                    #######################

                    ## 1. WILCOXON TEST ##
                    ## x=code and y=code+warnings
                    ## H0: x>y the performance (based on a metric) of the model using code features is greater than the performance when using code+warnings
                    ## if p-value <= 0.05, we reject the H0.
                    ## Note: to handle ties (x=y), we use the zsplit method
                    _, wilcoxon_p = wilcoxon(x, y, alternative='greater', zero_method='zsplit')

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
                    delta_cliffs, res_cliffs = cliffs_delta(x, y)

                    ## write results to csv
                    overall_test_results['metric'] = 'diff_' + f'{metric}'
                    overall_test_results['wilcoxon_test(p-value)'] = wilcoxon_p
                    overall_test_results['cliffs_delta'] = delta_cliffs
                    overall_test_results['cliffs_delta_result'] = res_cliffs

                    with open(ROOT_PATH + 'results/final_statistics.csv', 'a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=overall_test_results.keys())
                        writer.writerow(overall_test_results) 

                with open(ROOT_PATH + 'results/mean_median_overall_results.csv', 'a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
                    writer.writerow(overall_result_header)

                         

                ## draw histogram - to check the normality for diff_f1, diff_precision, diff_recall
                # columns = ['diff_precision','diff_recall','diff_f1', 'diff_auc']
                # for column in columns:
                #     hist = overall_df.hist(column=column)
                #     ## create histograms folder if it doesn't exist in results directory
                #     if not os.path.exists(ROOT_PATH + 'results/results-' + experiment_id + '/histograms'):
                #         os.makedirs(ROOT_PATH + 'results/results-' + experiment_id + '/histograms')
                #     plt.savefig(ROOT_PATH + 'results/results-' + experiment_id + '/histograms/final_hist_' + column +'_'+ str(SMOTE_TAG) + '.png')  




        

        # # draw histogram for all the input features in a single plot
        # df_raw_features = pd.read_csv(ROOT_PATH + 'data/understandability_with_warnings.csv')
        # with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
        #     experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
        #     for experiment in experiments:
        #         full_dataset = df_raw_features.dropna(subset=experiment["target"])
        #         feature_X_c = full_dataset[experiment["features"]].iloc[:, 1:]
        #         ## draw histogram with labels
        #         names = list(feature_X_c.columns)
        #         feature_X_c.hist(figsize=(10, 6), bins=len(feature_X_c.columns),  stacked=True)
        #         plt.suptitle('')
        #         plt.tight_layout()
        #         plt.savefig(ROOT_PATH + 'results/results-' + experiment_id + '/histograms/final_hist_code_features'+ experiment["target"] +'_'+ str(SMOTE_TAG) + '.png')
        #         plt.close()
