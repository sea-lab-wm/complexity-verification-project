import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import csv
import json

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

header = {'model':'', 'target':'', 
          'f1_c (median)':'', 'f1_cw (median)':'', 
          'precision_c (median)':'', 'precision_cw (median)':'',
          'recall_c (median)':'', 'recall_cw (median)':'', 
          'auc_c (median)':'', 'auc_cw (median)':'',
          'f1_c (mean)':'', 'f1_cw (mean)':'',
          'precision_c (mean)':'', 'precision_cw (mean)':'',
          'recall_c (mean)':'', 'recall_cw (mean)':'',
          'auc_c (mean)':'', 'auc_cw (mean)':''}
## write header
with open('ml_model/src/main/model/Results/mean_median_NO_SMOTE.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=header.keys())
    writer.writeheader()


for model_name in models:
    for target in targets:
        model_data = overall_df.query("model == '" + model_name + "' and target == '" + target + "'")
        
        header['model'] = model_name
        header['target'] = target

        ## number of data points
        ## we consider both hyperparameters obtained for code and code+warnings
        number_of_best_hyperparameters = model_data.shape[0]

        for metric in metrics.keys():
            ## compute the mean and median values for metrics for both code and code+warnings
            c_metric = metrics[metric][0]
            cw_metric = metrics[metric][1]
            metric_c_mean = model_data[c_metric].mean() ## since we don't have outliers we can reliably use the mean
            metric_cw_mean = model_data[cw_metric].mean()
            
            ## median values
            print(model_data.groupby(['model','target'])[metrics[metric]].median())
            header[f'{c_metric} (median)'] = model_data.groupby(['model','target'])[c_metric].median().values[0]
            header[f'{cw_metric} (median)'] = model_data.groupby(['model','target'])[cw_metric].median().values[0]
            
            ## mean values
            print(model_data.groupby(['model','target'])[metrics[metric]].mean())
            header[f'{c_metric} (mean)'] = model_data.groupby(['model','target'])[c_metric].mean().values[0]
            header[f'{cw_metric} (mean)'] = model_data.groupby(['model','target'])[cw_metric].mean().values[0]

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

        with open('ml_model/src/main/model/Results/mean_median_NO_SMOTE.csv', 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header.keys())
            writer.writerow(header)


## draw histogram - to check the normality for diff_f1, diff_precision, diff_recall
columns = ['diff_precision','diff_recall','diff_f1', 'diff_auc']
for column in columns:
    hist = overall_df.hist(column=column)
    plt.savefig('ml_model/src/main/model/Results/histograms/final_hist_' + column +'_NO_SMOTE.png')

## draw histogram for all the input features in a single plot
df_raw_features = pd.read_csv('ml_model/src/main/model/data/understandability_with_warnings.csv')
with open(ROOT_PATH + "classification/experiments.jsonl") as jsonl_file:
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
tests = {'metric':'', 'model':'', 'target':'', 'wilcoxon_test(p-value)':''}
## write header
with open('ml_model/src/main/model/Results/final_statistics_NO_SMOTE.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=tests.keys())
    writer.writeheader()

metrics = {'diff_f1': ['f1_c','f1_cw'], 
           'diff_precision': ['precision_c','precision_cw'], 
           'diff_recall': ['recall_c','recall_cw'],
           'diff_auc': ['auc_c','auc_cw']}

model_names = ["SVC", "knn_classifier", "logistic_regression", "randomForest_classifier", "mlp_classifier"]
targets = ['ABU50', 'BD50', 'PBU']

for model_name in models:
    for target in targets:
        model_data = overall_df.query("model == '" + model_name + "' and target == '" + target + "'")
        print (model_data.shape[0])
        for metric in metrics.keys():
            x = model_data[metrics[metric][0]] ## code
            y = model_data[metrics[metric][1]] ## code+warnings

            ## 1. WILCOXON TEST ##
            ## null hypothesis: There is no statistically significant difference between x and y
            ## if p-value <= 0.05, we reject the null hypothesis. i.e. there is a statistically significant difference between x and y
            ## Note: to handle ties (x=y), we use the zsplit method
            _, wilcoxon_p = wilcoxon(x, y, alternative='greater', zero_method='zsplit')

            ## write results to csv
            tests['metric'] = metric
            tests['model'] = model_name
            tests['target'] = target
            tests['wilcoxon_test(p-value)'] = wilcoxon_p

            with open('ml_model/src/main/model/Results/final_statistics_NO_SMOTE.csv', 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=tests.keys())
                writer.writerow(tests)