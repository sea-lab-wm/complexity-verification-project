'''
This file is used to process the raw_results from the experiments 
and generate the overall results, boxplots ,and statistical tests.
'''

import pandas as pd

import json
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta

ROOT_PATH = "/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"
df = pd.read_csv(ROOT_PATH + 'results/raw_results.csv')

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:

    overall_result_header = {
        'experiment':'', 
        'warning_feature':'', 
        'feature_set':'',
        '#best_hyperparameters':'', 
        'use_smote':'', 
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
        }
    
    ## write header
    with open(ROOT_PATH + 'results/mean_median_overall_results_new.csv', "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
        writer.writeheader()

    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

    for experiment in experiments:
        
        ## experiment id
        experiment_id = experiment["experiment_id"].split("-")[0] ## eg: exp1
        use_SMOTE = experiment["use_SMOTE"] ## True or False
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
        if not os.path.exists(ROOT_PATH + 'results/all-results/results-'+ experiment_id):
            os.makedirs(ROOT_PATH + 'results/all-results/results-'+ experiment_id)

        ## save the results
        overall_df.to_csv(ROOT_PATH + 'results/all-results/results-'+ experiment_id + '/overall_results_'+ str(SMOTE_TAG) + '.csv', index=False)

        # Group the data by target and find the rows with the maximum F1-scores for each target 
        best_code_features = overall_df.loc[overall_df.groupby('target')['f1_c'].idxmax()]
        best_code_warning_features = overall_df.loc[overall_df.groupby('target')['f1_cw'].idxmax()]
        
        # Write the results to a CSV file
        best_code_features.to_csv(ROOT_PATH + 'results/all-results/results-'+ experiment_id +'/best_code_models_'+ str(SMOTE_TAG) + '.csv', index=False)
        best_code_warning_features.to_csv(ROOT_PATH + 'results/all-results/results-'+ experiment_id + '/best_code+warning_models_'+ str(SMOTE_TAG) + '.csv', index=False)

        target = experiment["target"]

        metrics = {'f1': ['f1_c','f1_cw'],'precision': ['precision_c','precision_cw'],
                'recall': ['recall_c','recall_cw'], 'auc': ['auc_c','auc_cw']}
        
        models = ['SVC', 'knn_classifier', 'logistic_regression', 'randomForest_classifier', 'mlp_classifier', 'bayes_network']

        model_box_plot_image_names = [] 

        for model_name in models:
            model_data = overall_df.query("model == '" + model_name + "' and target == '" + target + "'")

            for metric in metrics.keys():
                ## remove nan values. This is because both code and code+warnings should have values to compute the difference
                model_data = model_data.dropna(subset=[metrics[metric][0], metrics[metric][1]])

                ## #data points
                number_of_best_hyperparameters = model_data.shape[0]

                ## get the code and code+warning metrics
                c_metric = metrics[metric][0] ## eg. f1_c
                cw_metric = metrics[metric][1]


                ## compute the mean of each metric
                metric_c_mean = model_data[c_metric].mean() ## mean of code
                metric_c_median = model_data[c_metric].median()

                ## compute the median of each metric
                metric_cw_mean = model_data[cw_metric].mean() ## mean of code+warning
                metric_cw_median = model_data[cw_metric].median()

                ## compute the mean and median difference between code and code+warning
                diff_mean = metric_cw_mean - metric_c_mean
                diff_median = metric_cw_median - metric_c_median

                #######################
                ## STATISTICAL TESTS ##
                #######################
                feature_set = experiment['experiment_id'].split('-')[3] ## eg: set1
                smote_value = experiment['use_SMOTE']
                warning_feature = experiment['features'][0]
                model = model_name
                specific_model_data_wilcoxon = df.query("feature_set == '" + feature_set + "' and iteration == 'overall' and target == '" + target + "'" + " and use_smote == " + str(smote_value) + " and warning_feature =='" + warning_feature + "' and model == '" + model + "'")

                x = specific_model_data_wilcoxon[c_metric] ## code data
                y = specific_model_data_wilcoxon[cw_metric] ## code+warnings data

                ## 1. WILCOXON TEST ##
                ## x=code and y=code+warnings
                ## H0: x >= y the performance (based on a metric) of the model using code 
                #  features is greater than the performance when using code+warnings
                ## if p-value <= 0.05, we reject the H0 this is doing alternative='less'
                ## Note: to handle ties (x=y), we use the zsplit method
                _, wilcoxon_p = wilcoxon(x, y, alternative='less', zero_method='zsplit')

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

                overall_result_header['experiment'] = experiment["experiment_id"]
                overall_result_header['warning_feature'] = experiment["features"][0]
                overall_result_header['feature_set'] = experiment["experiment_id"].split("-")[3] ## eg: set1
                overall_result_header['#best_hyperparameters'] = number_of_best_hyperparameters
                overall_result_header['use_smote'] = use_SMOTE
                overall_result_header['model'] = model_name
                overall_result_header['target'] = target
                overall_result_header['metric'] = metric
                overall_result_header['code(median)'] = metric_c_median
                overall_result_header['code+warning(median)'] = metric_cw_median
                overall_result_header['code(mean)'] = metric_c_mean
                overall_result_header['code+warning(mean)'] = metric_cw_mean
                overall_result_header['diff(median)'] = diff_median
                overall_result_header['diff(mean)'] = diff_mean
                overall_result_header['wilcoxon_test(p-value)'] = wilcoxon_p
                overall_result_header['cliffs_delta'] = delta_cliffs
                overall_result_header['cliffs_delta_result'] = res_cliffs

                with open(ROOT_PATH + 'results/mean_median_overall_results_new.csv', "a") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=overall_result_header.keys())
                    writer.writerow(overall_result_header)


                ###############
                ## BOX PLOTS ##
                ###############
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
                if not os.path.exists(ROOT_PATH + 'results/all-results/results-' + experiment_id + '/final-boxplots'):
                    os.makedirs(ROOT_PATH + 'results/all-results/results-' + experiment_id + '/final-boxplots')
                plt.savefig(ROOT_PATH + 'results/all-results/results-' + experiment_id + '/final-boxplots/' + target + '_' + model_name + '_' + metric +'_boxplot_'+ str(SMOTE_TAG) + '.png')
                plt.clf()
                plt.close() 

                if metric == 'f1': ## we will combine all the f1 boxplots for each model
                    model_box_plot_image_names.append(ROOT_PATH + 'results/all-results/results-' + experiment_id + '/final-boxplots/' + target + '_' + model_name + '_' + metric +'_boxplot_'+ str(SMOTE_TAG) + '.png')
        
        ## combine all the f1 boxplots for each model to one image
        plt.figure(figsize = (10,6))
        gs1 = gridspec.GridSpec(3, 3) ## 
        gs1.update(wspace=0.00005, hspace=0.00005) # set the spacing between axes.
        for i, image in zip(range(6), model_box_plot_image_names):
            ax = plt.subplot(gs1[i])
            plt.axis('off')
            ax.imshow(plt.imread(image))
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(ROOT_PATH + 'results/all-results/results-' + experiment_id + '/final_boxplots_' + experiment_id + '_f1.png', dpi=900, bbox_inches='tight')
        plt.clf()
        plt.close()