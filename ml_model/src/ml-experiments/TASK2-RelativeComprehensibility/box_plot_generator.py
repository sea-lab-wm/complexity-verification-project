import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats

ROOT_PATH="/home/nadeeshan/VPro2/complexity-verification-project/ml_model/src/ml-experiments"

DS3_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/DS3_TASK1_FINAL.csv"
DS3_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/DS3_TASK2_FINAL.csv"

# DS6_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/DS6_TASK1_SVC_Standard.csv"
# DS6_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/DS6_TASK2_SVC_Standard.csv"
DS6_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/results/DS6_TASK1_Robust.csv"
DS6_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/DS6_TASK2_Robust.csv"

MERGED_DS_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/MERGED_DS_TASK2_NEW.csv"

ds6_targets = ["ABU", "ABU50", "BD", "BD50", "PBU", "AU"]
ds3_targets = ["readability_level"]

def boxplot_generator(df_task1, df_task2, title, save_path):
    
    # Get unique model names across both tasks
    unique_models = sorted(set(df_task1['model'].unique()) | set(df_task2['model'].unique()))
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Offset for second boxplot to appear side by side
    positions_task1 = range(1, len(unique_models) + 1)
    positions_task2 = [p + 0.4 for p in positions_task1]  # Slightly offset positions

    # Mean circle properties
    meanprops = dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8)
    num_task1_models = ""
    num_task2_models = ""

    # Iterate over the unique models and create box plots if data exists for that model
    for pos1, pos2, model in zip(positions_task1, positions_task2, unique_models):
        # Check if the model exists in task 1 and task 2
        if model in df_task1['model'].unique():
            filtered_df_task1_model = df_task1[df_task1['model'] == model].copy()
            
            filtered_df_task1_model["RI=(F1_weighted-baseline)/baseline"] *= 100  # Convert to percentage
            filtered_df_task1_model.boxplot(column=["RI=(F1_weighted-baseline)/baseline"], ax=ax, 
                                            positions=[pos1], widths=0.35, patch_artist=True, 
                                            boxprops=dict(facecolor='blue'), showmeans=True, meanprops=meanprops)
            
            ## get the number of data points in each box plot ##
            for model in filtered_df_task1_model["model"].unique():
                plt.suptitle("Number of data points in each box plot" , fontsize=10)
                n = filtered_df_task1_model.query("model == @model").shape[0]
                num_task1_models+= model + ": " + str(n) + "  "
            

        if model in df_task2['model'].unique():
            filtered_df_task2_model = df_task2[df_task2['model'] == model].copy()

            filtered_df_task2_model["RI=(F1_weighted-baseline)/baseline"] *= 100   # Convert to percentage            
            filtered_df_task2_model.boxplot(column=["RI=(F1_weighted-baseline)/baseline"], ax=ax, 
                                            positions=[pos2], widths=0.35, patch_artist=True, 
                                            boxprops=dict(facecolor='orange'), showmeans=True, meanprops=meanprops)
      
            

            ## get the number of data points in each box plot ##
            for model in filtered_df_task2_model["model"].unique():
                plt.suptitle("Number of data points in each box plot" , fontsize=10)
                n = filtered_df_task2_model.query("model == @model").shape[0]
                num_task2_models+= model + ": " + str(n) + "  "
            

    plt.suptitle("TASK1 - " + num_task1_models + '\n' + "TASK2 - " + num_task2_models, fontsize=10)
        

    # Rotate x-axis labels (model names) by 8 degrees
    plt.xticks(positions_task1, unique_models, rotation=8, ha="right")

    ax.set_title(title)
    ax.set_ylabel("RI = (F1_weighted - baseline)/baseline (%)")
    ax.set_xlabel("Model")

    # Format y-axis values as percentages
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # Add legend to distinguish between Task 1 and Task 2
    handles = [plt.Line2D([0], [0], color='blue', lw=2, label='Task 1'),
            plt.Line2D([0], [0], color='orange', lw=2, label='Task 2'),
            plt.Line2D([0], [0], marker='o', color='red', label='Mean')]
    ax.legend(handles=handles)

    # Save the plot
    ## if folder not exists create it ##
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

def f1_weighted_baseline_results(df_task1, df_task2, dataset, avg, option):
    """
    df_task1: DataFrame containing the results of Task 1 (non-filtered)
    df_task2: DataFrame containing the results of Task 2 (non-filtered)
    dataset: Name of the dataset (e.g. DS3, DS6)
    avg: Boolean indicating whether to calculate the average or median
    option: Type of results to calculate (GrandTotal, F1_weighted_lte_baseline, F1_weighted_gt_baseline)
    """
    csv_header = {"Dataset": "", "Target":""}
    
    ## Get unique model names across both tasks
    unique_models = sorted(set(df_task1['model'].unique()) | set(df_task2['model'].unique()))
    
    ## add dynamic columns to the csv header ##
    for model in unique_models:
        csv_header[model + "_(%_of_models)_(Task1)"] = ""
        csv_header[model + "_(%_of_models)_(Task2)"] = ""
        csv_header[model + "_(%_of_models)_(Task2-Task1)"] = ""
        csv_header[model + "_(RI)_(Task1)"] = ""
        csv_header[model + "_(RI)_(Task2)"] = ""
        csv_header[model + "_(RI)_(Task2-Task1)"] = ""


    if option == "F1_weighted_gt_baseline":
        ## F1_weighted > baseline ##
        filtered_df_task1 = df_task1.query("`F1_weighted > baseline` == 1")
        filtered_df_task2 = df_task2.query("`F1_weighted > baseline` == 1")
    elif option == "F1_weighted_lte_baseline":
        ## F1_weighted <= baseline ##
        filtered_df_task1 = df_task1.query("`F1_weighted > baseline` == 0")
        filtered_df_task2 = df_task2.query("`F1_weighted > baseline` == 0")
    else:
        ## Grand Total ##
        filtered_df_task1 = df_task1
        filtered_df_task2 = df_task2

   
    for model in unique_models:
        ## % of models ##
        percent_of_models_task1 = filtered_df_task1.query('model == @model').shape[0] / df_task1.query('model == @model').shape[0] * 100 if df_task1.query('model == @model').shape[0] > 0 else 0
        percent_of_models_task2 = filtered_df_task2.query('model == @model').shape[0] / df_task2.query('model == @model').shape[0] * 100 if df_task2.query('model == @model').shape[0] > 0 else 0

        csv_header[model + "_(%_of_models)_(Task1)"] = percent_of_models_task1
        csv_header[model + "_(%_of_models)_(Task2)"] = percent_of_models_task2

        if avg:
            title="Average"
            ## avg RI ##
            mean_ri_task1 = filtered_df_task1.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].mean() if len(filtered_df_task1.query('model == @model')) > 0 else 0
            mean_ri_task2 = filtered_df_task2.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].mean() if len(filtered_df_task2.query('model == @model')) > 0 else 0
            csv_header[model + "_(RI)_(Task1)"] = mean_ri_task1
            csv_header[model + "_(RI)_(Task2)"] = mean_ri_task2
            csv_header[model + "_(RI)_(Task2-Task1)"] = (mean_ri_task2 - mean_ri_task1)
        else:
            title="Median"
            ## median RI ##    
            median_ri_task1 = filtered_df_task1.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].median() if len(filtered_df_task1.query('model == @model')) > 0 else 0
            median_ri_task2 = filtered_df_task2.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].median() if len(filtered_df_task2.query('model == @model')) > 0 else 0
            csv_header[model + "_(RI)_(Task1)"] = median_ri_task1
            csv_header[model + "_(RI)_(Task2)"] = median_ri_task2
            csv_header[model + "_(RI)_(Task2-Task1)"] = median_ri_task2 - median_ri_task1

        
        csv_header[model + "_(%_of_models)_(Task2-Task1)"] = percent_of_models_task2 - percent_of_models_task1
        if option == "F1_weighted_lte_baseline":
            csv_header[model + "_(%_of_models)_(Task2-Task1)"] = percent_of_models_task1 - percent_of_models_task2

    csv_header["Dataset"] = dataset
    csv_header["Target"] = df_task1["target"].unique()[0]  

    ## if file exists, append it, if not create ##
    if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/F1_weighted_" + title + "_" + option + "_table.csv"):
        with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/F1_weighted_" + title + "_" + option + "_table.csv", 'a') as f:
            w = csv.DictWriter(f, csv_header.keys())
            w.writerow(csv_header)
    else:
        with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/F1_weighted_" + title + "_" + option + "_table.csv", 'w+') as f:
            w = csv.DictWriter(f, csv_header.keys())
            w.writeheader()
            w.writerow(csv_header)

def statistics_results(df_task1, df_task2, dataset, option):
    csv_header = {"Dataset": "", "Target":""}
    
    ## Get unique model names across both tasks
    unique_models = sorted(set(df_task1['model'].unique()) | set(df_task2['model'].unique()))
    
    ## add dynamic columns to the csv header ##
    for model in unique_models:
        csv_header[model + "_(# of models)(Task1)"] = ""
        csv_header[model + "_(# of models)(Task2)"] = ""
        csv_header[model + "_(p-value)"] = ""
        csv_header[model + "_(statistic)"] = ""
        csv_header[model + "_(% confidence level)"] = ""
        csv_header[model + "_(Interpretation)"] = ""

    if option == "F1_weighted_gt_baseline":
        ## F1_weighted > baseline ##
        filtered_df_task1 = df_task1.query("`F1_weighted > baseline` == 1")
        filtered_df_task2 = df_task2.query("`F1_weighted > baseline` == 1")
    elif option == "F1_weighted_lte_baseline":
        ## F1_weighted <= baseline ##
        filtered_df_task1 = df_task1.query("`F1_weighted > baseline` == 0")
        filtered_df_task2 = df_task2.query("`F1_weighted > baseline` == 0")
    else:
        ## Grand Total ##
        filtered_df_task1 = df_task1
        filtered_df_task2 = df_task2

    for model in unique_models:
        ## # of models ##
        num_of_models_task1 = filtered_df_task1.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].shape[0]
        num_of_models_task2 = filtered_df_task2.query('model == @model')['RI=(F1_weighted-baseline)/baseline'].shape[0]

        csv_header[model + "_(# of models)(Task1)"] = num_of_models_task1
        csv_header[model + "_(# of models)(Task2)"] = num_of_models_task2

        ## perform mann-whitney U test ##
        ## We perform this to know whether the distributions of the RI(Task1) and RI(Task2) are significantly different ##
        ## This is a non-parametric test (i.e does not assume normal distribution of samples) ##
        ## Null hypothesis (H0): There is no difference between the ranks of the RI(Task1) and RI(Task2) i.e the distribution underlying sample x is the same as the distribution underlying sample y ##
        ## Alternative hypothesis (Ha): RI(Task1) has lower ranks than RI(Task2) ##
        ## If p-value <= 0.05, we reject the null hypothesis ##
        statistic, p_value = stats.mannwhitneyu(filtered_df_task1.query('model == @model')['RI=(F1_weighted-baseline)/baseline'], 
                                                filtered_df_task2.query('model == @model')['RI=(F1_weighted-baseline)/baseline'], alternative='less', method='exact')
        
        csv_header[model + "_(Interpretation)"] = "Ha: RI(Task1) < RI(Task2)" if p_value <= 0.05 else "H0: RI(Task1) >= RI(Task2)"
        csv_header[model + "_(% confidence level)"] = 100 - p_value * 100 if not np.isnan(p_value) else "N/A"
        if np.isnan(p_value):
            p_value = statistic = "N/A"
            csv_header[model + "_(Interpretation)"] = "N/A"
            csv_header[model + "_(% confidence level)"] = "N/A"
            
        csv_header[model + "_(p-value)"] = p_value
        csv_header[model + "_(statistic)"] = statistic
        

        
    csv_header["Dataset"] = dataset
    csv_header["Target"] = df_task1["target"].unique()[0]

    ## if file exists, append it, if not create ##
    if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/statistics_" + option + "_table.csv"):
        with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/statistics_" + option + "_table.csv", 'a') as f:
            w = csv.DictWriter(f, csv_header.keys())
            w.writerow(csv_header)
    else:
        with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/statistics_" + option + "_table.csv", 'w+') as f:
            w = csv.DictWriter(f, csv_header.keys())
            w.writeheader()
            w.writerow(csv_header)


def main():
    DS3_TASK1 = pd.read_csv(DS3_TASK1_PATH)
    DS3_TASK2 = pd.read_csv(DS3_TASK2_PATH)

    DS6_TASK1 = pd.read_csv(DS6_TASK1_PATH)
    DS6_TASK2 = pd.read_csv(DS6_TASK2_PATH)


    ## DS3_TASK1 vs DS3_TASK2 boxplot ##
    # for target in ds3_targets:
        
    #         iteration_task2 = "overall"
    #         iteration_task1 = "Overall"


    #         ## Grand Total ##
    #         ## Filter data for task 1 and task 2 ##
    #         filtered_df_task1 = DS3_TASK1.query("target == @target & iteration == @iteration_task1").sort_values(by="model")
    #         filtered_df_task2 = DS3_TASK2.query("target == @target & iteration == @iteration_task2").sort_values(by="model")
    #         plot_title = "DS3-Task1 & DS3-Task2 (Target = " + target  + ", Grand Total)"
    #         plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS3_TASK1_TASK2_" + target + "_GrandTotal.png"
    #         ## generate the box plot ##
    #         boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)

    #         ## Generate the table of results ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", True, "GrandTotal")## Grand Total Avg ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", True, "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", True, "F1_weighted_gt_baseline")## F1_weighted > baseline ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", False, "GrandTotal")## Grand Total Median ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", False, "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
    #         f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS3", False, "F1_weighted_gt_baseline")## F1_weighted > baseline ##

    #         ## statistics results ##
    #         statistics_results(filtered_df_task1, filtered_df_task2, "DS3", "GrandTotal")## Grand Total ##
    #         statistics_results(filtered_df_task1, filtered_df_task2, "DS3", "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
    #         statistics_results(filtered_df_task1, filtered_df_task2, "DS3", "F1_weighted_gt_baseline")## F1_weighted > baseline ##

    #         ## F1_weighted > baseline ##
    #         filtered_df_task1 = DS3_TASK1.query("target == @target & iteration == @iteration_task1 & `F1_weighted > baseline` == 1").sort_values(by="model")
    #         filtered_df_task2 = DS3_TASK2.query("target == @target & iteration == @iteration_task2 & `F1_weighted > baseline` == 1").sort_values(by="model")
    #         plot_title = "DS3-Task1 & DS3-Task2 (Target = " + target + ", F1_weighted > baseline)"  
    #         plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS3_TASK1_TASK2_" + target + "_F1_weighted_gt_baseline.png"
    #         boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)
            
    #         ## F1_weighted <= baseline ##
    #         filtered_df_task1 = DS3_TASK1.query("target == @target & iteration == @iteration_task1 & `F1_weighted > baseline` == 0").sort_values(by="model")
    #         filtered_df_task2 = DS3_TASK2.query("target == @target & iteration == @iteration_task2 & `F1_weighted > baseline` == 0").sort_values(by="model")
    #         plot_title = "DS3-Task1 & DS3-Task2 (Target = " + target + ", F1_weighted <= baseline)"
    #         plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS3_TASK1_TASK2_" + target + "_F1_weighted_lte_baseline.png"
    #         boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)




    ## DS6_TASK1 vs DS6_TASK2 boxplot - Grand Total ##
    for target in ds6_targets:
        
            iteration_task2 = "overall"
            iteration_task1 = "Overall"
            
            
            ## Grand Total ##
            # Filter data for task 1 and task 2
            filtered_df_task1 = DS6_TASK1.query("target == @target  & iteration == @iteration_task1").sort_values(by="model")
            filtered_df_task2 = DS6_TASK2.query("target == @target  & iteration == @iteration_task2").sort_values(by="model")
            
            plot_title = "DS6-Task1 & DS6-Task2 (Target = " + target  + ", Grand Total)"
            plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS6_TASK1_TASK2_" + target + "_GrandTotal.png"
            ## generate the box plot ##
            boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)

            ## Generate the table of results ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", True, "GrandTotal")## Grand Total Avg ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", True, "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", True, "F1_weighted_gt_baseline")## F1_weighted > baseline ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", False, "GrandTotal")## Grand Total Median ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", False, "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
            f1_weighted_baseline_results(filtered_df_task1, filtered_df_task2, "DS6", False, "F1_weighted_gt_baseline")## F1_weighted > baseline ##

            ## statistics results ##
            statistics_results(filtered_df_task1, filtered_df_task2, "DS6", "GrandTotal")## Grand Total ##
            statistics_results(filtered_df_task1, filtered_df_task2, "DS6", "F1_weighted_lte_baseline")## F1_weighted <= baseline ##
            statistics_results(filtered_df_task1, filtered_df_task2, "DS6", "F1_weighted_gt_baseline")## F1_weighted > baseline ##

            ## F1_weighted > baseline ##
            filtered_df_task1 = DS6_TASK1.query("target == @target  & iteration == @iteration_task1 & `F1_weighted > baseline` == 1").sort_values(by="model")
            filtered_df_task2 = DS6_TASK2.query("target == @target  & iteration == @iteration_task2 & `F1_weighted > baseline` == 1").sort_values(by="model")
            
            plot_title = "DS6-Task1 & DS6-Task2 (Target = " + target + ", F1_weighted > baseline)" 
            plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS6_TASK1_TASK2_" + target + "_F1_weighted_gt_baseline.png"

            boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)

            ## F1_weighted <= baseline ##
            filtered_df_task1 = DS6_TASK1.query("target == @target  & iteration == @iteration_task1 & `F1_weighted > baseline` == 0").sort_values(by="model")
            filtered_df_task2 = DS6_TASK2.query("target == @target  & iteration == @iteration_task2 & `F1_weighted > baseline` == 0").sort_values(by="model")

            plot_title = "DS6-Task1 & DS6-Task2 (Target = " + target + ", F1_weighted <= baseline)"
            plot_path = ROOT_PATH + "/TASK2-RelativeComprehensibility/results/Test/box_plot_DS6_TASK1_TASK2_" + target + "_F1_weighted_lte_baseline.png"

            boxplot_generator(filtered_df_task1, filtered_df_task2, plot_title, plot_path)

    # # ## MERGED_DS_TASK2_NEW target_wise boxplot ##
    # for x in DS3_TASK2["F1_weighted > baseline"].unique():
    #     iteration = "overall"
    #     filtered_df = DS3_TASK2.query("`F1_weighted > baseline` == @x")
    #     for target in ds3_targets:
    #         filtered_df_target_model = filtered_df.query("target == @target & iteration == @iteration").sort_values(by="model")
            
    #         fig = plt.figure(figsize =(10, 5))
    #         filtered_df_target_model.boxplot(column=["RI=(F1_weighted-baseline)/baseline"], by=["model"], ax=fig.gca())

    #         if x == 0:
    #             plt.title("DS3-Task2 (F1_weighted <= baseline)")
    #         else:
    #             plt.title("DS3-Task2 (F1_weighted > baseline)")
            
    #         plt.ylabel("RI=(F1_weighted - baseline)/baseline")
    #         plt.xlabel("Model")
    #         text = ""
    #         for model in filtered_df_target_model["model"].unique():
    #             plt.suptitle( "Number of data points in each box plot" , fontsize=10)
    #             n = filtered_df_target_model.query("model == @model").shape[0]
    #             text+=str(n) + "  "
    #         plt.suptitle(text, fontsize=10)
    #         plt.savefig(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/boxplot/target_wise/box_plot_DS3_TASK2_" + target + "_RI_" + str(x) + ".png")
    #         plt.close()

    # ## MERGED_DS_TASK2_NEW target_wise boxplot ## 
    # iteration = "overall"
    # for target in ds3_targets:
    #     filtered_df_target_model = filtered_df.query("target == @target & iteration == @iteration").sort_values(by="model")
        
    #     fig = plt.figure(figsize =(10, 5))
    #     filtered_df_target_model.boxplot(column=["RI=(F1_weighted-baseline)/baseline"], by=["model"], ax=fig.gca())

        
    #     plt.title("MERGED-Task2 (GradTotal)")
        
    #     plt.ylabel("RI=(F1_weighted - baseline)/baseline")
    #     plt.xlabel("Model")
    #     text = ""
    #     for model in filtered_df_target_model["model"].unique():
    #         plt.suptitle( "Number of data points in each box plot" , fontsize=10)
    #         n = filtered_df_target_model.query("model == @model").shape[0]
    #         text+=str(n) + "  "
    #     plt.suptitle(text, fontsize=10)
    #     plt.savefig(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/boxplot/target_wise/box_plot_MERGED_DS_TASK2_" + target + "_RI_" + str(x) + ".png")
    #     plt.close()                        

if __name__ == "__main__":
    ## delete existing files ##
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_GrandTotal_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_GrandTotal_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_F1_weighted_lte_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_F1_weighted_lte_baseline_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_F1_weighted_gt_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Average_F1_weighted_gt_baseline_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_GrandTotal_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_GrandTotal_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_F1_weighted_lte_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_F1_weighted_lte_baseline_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_F1_weighted_gt_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/F1_weighted_Median_F1_weighted_gt_baseline_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_GrandTotal_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_GrandTotal_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_F1_weighted_gt_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_F1_weighted_gt_baseline_table.csv")
    # if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_F1_weighted_lte_baseline_table.csv"):
    #     os.remove(ROOT_PATH + "/TASK2-RelativeComprehensibility/results/statistics_F1_weighted_lte_baseline_table.csv")
    main()            