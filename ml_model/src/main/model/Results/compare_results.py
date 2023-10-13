import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import csv
import json
import os

from scipy.stats import wilcoxon
from scipy.stats import normaltest


df = pd.read_csv('ml_model/src/main/model/Results/raw_results_NO_SMOTE.csv')

ROOT_PATH = "ml_model/src/main/model/"

############################
## FILTER OVERALL RESULTS ##
############################
## filter overall results
overall_df = df.query("iteration == 'overall' and use_smote == False")
## remove duplicate rows ##
# overall_df = overall_df.drop_duplicates(subset=['target','tp_c','tn_c','fp_c','fn_c','tp_cw','tn_cw','fp_cw','fn_cw','precision_c','recall_c','f1_c','precision_cw','recall_cw','f1_cw'], keep='first')
## save overall results
overall_df.to_csv('ml_model/src/main/model/Results/overall_results_NO_SMOTE.csv', index=False)
stats = overall_df[['f1_c','f1_cw']].describe()

# Group the data by target and find the rows with the maximum F1-scores for each target 
best_code_features = overall_df.loc[overall_df.groupby('target')['f1_c'].idxmax()]
best_code_warning_features = overall_df.loc[overall_df.groupby('target')['f1_cw'].idxmax()]
# Write the results to a CSV file
best_code_features.to_csv('ml_model/src/main/model/Results/best_code_models_NO_SMOTE.csv', index=False)
best_code_warning_features.to_csv('ml_model/src/main/model/Results/best_code+warning_models_NO_SMOTE.csv', index=False)


##############
## BOX PLOTS ##
##############
targets = ['ABU50', 'BD50', 'PBU']
metrics = {'f1': ['f1_c','f1_cw'],'precision': ['precision_c','precision_cw'],
           'recall': ['recall_c','recall_cw'], 'ROC_AUC': ['auc_c','auc_cw']}
models = ['SVC','knn_classifier', 'logistic_regression', 'randomForest_classifier', 'mlp_classifier']
for target in targets:
    for model_name in models:
        model_data = overall_df[overall_df['model'] == model_name]
        ## number of data points
        number_of_best_hyperparameters = model_data.shape[0]
        for metric in metrics.keys():
            ## compute the mean and median values for metrics for both code and code+warnings
            c_metric = metrics[metric][0]
            cw_metric = metrics[metric][1]
            metric_c_mean = model_data[c_metric].mean() ## since we don't have outliers we can reliably use the mean
            metric_cw_mean = model_data[cw_metric].mean()
            
            ## median values
            print(model_data.groupby(['model','target', 'experiment'])[metrics[metric]].median())
            model_data.groupby(['model','target', 'experiment'])[metrics[metric]].median().to_csv('ml_model/src/main/model/Results/median_values/' + model_name +'_'+ metric + '_median_NO_SMOTE.csv')
            
            ## mean values
            print(model_data.groupby(['model','target', 'experiment'])[metrics[metric]].mean())
            model_data.groupby(['model','target', 'experiment'])[metrics[metric]].mean().to_csv('ml_model/src/main/model/Results/mean_values/' + model_name +'_'+ metric + '_mean_NO_SMOTE.csv')

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
            plt.savefig('ml_model/src/main/model/Results/final-boxplots/' + target + '_' + model_name + '_' + metric +'_boxplot_NO_SMOTE.png')
            plt.close()  


## draw histogram - to check the normality for diff_f1, diff_precision, diff_recall
columns = ['diff_precision','diff_recall','diff_f1', 'diff_auc']
for column in columns:
    hist = overall_df.hist(column=column)
    plt.savefig('ml_model/src/main/model/Results/histograms/final_hist_' + column +'_NO_SMOTE.png')

## draw histogram for all the input features in a single plot
df_raw_features = pd.read_csv('ml_model/src/main/model/data/understandability_with_warnings.csv')
with open(ROOT_PATH + "ClassificationModels/experiments.jsonl") as jsonl_file:
    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]
    for experiment in experiments:
        full_dataset = df_raw_features.dropna(subset=experiment["target"])
        feature_X_c = full_dataset[experiment["features"]].iloc[:, 1:]
        ## draw histogram with labels
        names = list(feature_X_c.columns)
        feature_X_c.hist(figsize=(10, 6), bins=len(feature_X_c.columns),  stacked=True)
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('ml_model/src/main/model/Results/histograms/final_hist_code_features'+ experiment["target"] +'_NO_SMOTE.png')
        plt.close()

#######################
## STATISTICAL TESTS ##
#######################
tests = {'feature':'', 'wilcoxon_test':''}
## write header
with open('ml_model/src/main/model/Results/final_statistics_NO_SMOTE.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=tests.keys())
    writer.writeheader()

columns = {'diff_f1': ['f1_c','f1_cw'], 
           'diff_precision': ['precision_c','precision_cw'], 
           'diff_recall': ['recall_c','recall_cw'],
           'diff_auc': ['auc_c','auc_cw']}
for column in columns.keys():
    x = overall_df[columns[column][0]] ## code
    y = overall_df[columns[column][1]] ## code+warnings

    ## 1. WILCOXON TEST ##
    ## null hypothesis: x is greater than y
    ## if p-value < 0.05, we reject the null hypothesis
    _, wilcoxon_p = wilcoxon(x, y, alternative='greater')

    ## write results to csv
    tests['feature'] = column
    tests['wilcoxon_test'] = wilcoxon_p

    with open('ml_model/src/main/model/Results/final_statistics_NO_SMOTE.csv', 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tests.keys())
        writer.writerow(tests)
          

# def image_grid(dir):
#     model_names = ["SVC", "knn_classifier", "logistic_regression", "randomForest_classifier", "mlp_classifier"]
#     targets = ["PBU", "ABU50", "BD50"]    
#     for target in targets:
#         for model_name in model_names:
#             image_f1 = plt.imread(dir + target + '_' + model_name + '_f1_boxplot.png')
#             image_auc = plt.imread(dir + target + '_' + model_name + '_ROC_AUC_boxplot.png')
#             image_precision = plt.imread(dir + target + '_' + model_name + '_precision_boxplot.png')
#             image_recall = plt.imread(dir + target + '_' + model_name + '_recall_boxplot.png')
            

#             # Create a figure and axes
#             fig, axes = plt.subplots(nrows=2, ncols=2)

#             # Plot the images
#             axes[0, 0].imshow(image_f1)
#             axes[0, 1].imshow(image_auc)
#             axes[1, 0].imshow(image_precision)
#             axes[1, 1].imshow(image_recall)

#             # Set the title and labels
#             plt.title(target + '_' + model_name)
#             plt.xlabel("")
#             plt.ylabel("")

#             plt.savefig('ml_model/src/main/model/Results/final-boxplots/'+ target + '_' + model_name + '.png')
#             plt.close()

# image_grid('ml_model/src/main/model/Results/final-boxplots/')