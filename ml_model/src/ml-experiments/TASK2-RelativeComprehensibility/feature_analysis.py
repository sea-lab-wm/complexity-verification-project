import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

task1_features = ["#comments (avg)", "#nested blocks (avg)", "Extent of aligned blocks", "#conditionals (dft)", "#operators (dft)", "#aligned blocks", "Identifiers length (max)", "#keywords (max)", "#assignments (avg)", "#comparisons (dft)", "#operators (avg)", "Indentation length (avg)", "CICSyn (avg)", "CIC (avg)", "Readability", "Entropy", "#loops (dft)", "Indentation length (max)", "LOC", "#assignments", "Strings (Visual Y)", "#keywords (dft)", "#periods", "#assignments (dft)", "CIC (max)", "Numbers (Visual Y)", "Identifiers (Visual Y)", "#spaces", "#loops (avg)", "#parameters", "CICSyn (max)", "#numbers (dft)", "Numbers (Visual X)", "Operators (Visual Y)", "Keywords (Visual Y)", "Strings (Visual X)", "NMI (max)", "#periods (dft)", "Indentation length (dft)", "#spaces (dft)", "Comments (Visual Y)", "NM (max)", "NMI (avg)", "#identifiers (dft)", "Cyclomatic complexity", "#comments (dft)", "#parenthesis (avg)", "#periods (avg)", "#parenthesis (dft)", "#words (max)", "#blank lines (avg)", "Literals (Visual X)", "Literals (Visual Y)", "#conditionals (avg)", "#identifiers (min)", "#numbers (max)", "#statements", "#commas", "MIDQ (min)", "MIDQ (avg)", "MIDQ (max)", "#spaces (avg)", "#keywords (avg)", "#commas (avg)", "Comments (Visual X)", "ITID (min)", "#identifiers", "Line length (max)", "NM (avg)", "Line length (dft)", "#identifiers (max)", "Keywords (Visual X)", "#characters (max)", "Identifiers (Visual X)", "#numbers (avg)", "Operators (Visual X)", "Volume", "#comparisons (avg)", "#identifiers (avg)", "CR", "Identifiers Length (avg)", "#commas (dft)", "#literals", "ITID (avg)", "Line length (avg)"]
task2_features = ["#operators (dft)_y", "#operators (dft)_x", "#aligned blocks_y", "#aligned blocks_x", "#comparisons (dft)_y", "#comparisons (dft)_x", "#numbers (avg)_y", "#numbers (avg)_x", "Extent of aligned blocks_y", "Extent of aligned blocks_x", "#numbers (max)_y", "#numbers (max)_x", "#operators (avg)_y", "#operators (avg)_x", "ITID (min)_x", "ITID (min)_y", "#assignments (dft)_y", "#periods_y", "#assignments_x", "#assignments_y", "#assignments (dft)_x", "Identifiers (Visual Y)_x", "Identifiers (Visual Y)_y", "#nested blocks (avg)_x", "#nested blocks (avg)_y", "#periods_x", "Keywords (Visual Y)_x", "Keywords (Visual Y)_y", "#assignments (avg)_x", "Numbers (Visual X)_y", "#assignments (avg)_y", "Numbers (Visual X)_x", "#literals_y", "#literals_x", "#comments (dft)_y", "Comments (Visual Y)_y", "#comments (dft)_x", "#periods (avg)_y", "#parameters_x", "#parameters_y", "Comments (Visual Y)_x", "#comments (avg)_y", "Operators (Visual Y)_x", "Operators (Visual Y)_y", "#comments (avg)_x", "#comparisons (avg)_y", "Strings (Visual Y)_y", "#comparisons (avg)_x", "Strings (Visual Y)_x", "#periods (avg)_x", "CICSyn (avg)_y", "CICSyn (avg)_x", "Indentation length (avg)_y", "Literals (Visual Y)_x", "Literals (Visual Y)_y", "Line length (dft)_y", "Literals (Visual X)_x", "Literals (Visual X)_y", "Indentation length (avg)_x", "#loops (dft)_y", "Line length (dft)_x", "Volume_y", "#identifiers (min)_y", "#keywords (avg)_x", "#keywords (avg)_y", "#identifiers (min)_x", "Indentation length (max)_y", "#loops (dft)_x", "Volume_x", "Strings (Visual X)_y", "Indentation length (max)_x", "Strings (Visual X)_x", "Identifiers Length (avg)_y", "Identifiers Length (avg)_x", "CR_y", "CR_x", "Numbers (Visual Y)_y", "#keywords (max)_x", "CIC (avg)_y", "CIC (avg)_x", "#keywords (max)_y", "Numbers (Visual Y)_x", "Identifiers length (max)_x", "#loops (avg)_y", "Identifiers length (max)_y", "#blank lines (avg)_x", "#blank lines (avg)_y", "#periods (dft)_x", "#loops (avg)_x", "#spaces (avg)_y", "#commas (avg)_x", "#commas (avg)_y", "NM (avg)_y", "#periods (dft)_y", "#spaces (avg)_x", "NM (avg)_x", "#conditionals (dft)_x", "#identifiers_y", "#conditionals (dft)_y", "#spaces_y", "#commas_x", "#conditionals (avg)_x", "#commas_y", "#identifiers_x", "CICSyn (max)_x", "#conditionals (avg)_y", "CICSyn (max)_y", "#parenthesis (dft)_x", "#spaces_x", "#parenthesis (dft)_y", "ITID (avg)_y", "#identifiers (dft)_x", "Readability_x", "CIC (max)_x", "Indentation length (dft)_x", "#spaces (dft)_x", "ITID (avg)_x", "#statements_x", "#identifiers (dft)_y", "CIC (max)_y", "#spaces (dft)_y", "Indentation length (dft)_y", "Readability_y", "#statements_y", "#keywords (dft)_x", "Line length (avg)_x", "#keywords (dft)_y", "Entropy_y", "#parenthesis (avg)_y", "Line length (avg)_y", "NMI (avg)_y", "Entropy_x", "LOC_x", "#parenthesis (avg)_x", "LOC_y", "#numbers (dft)_x", "#numbers (dft)_y", "NMI (avg)_x", "#commas (dft)_x", "#words (max)_y", "Comments (Visual X)_x", "#commas (dft)_y", "Comments (Visual X)_y", "#identifiers (avg)_y", "NM (max)_y", "#words (max)_x", "#characters (max)_x", "MIDQ (max)_x", "MIDQ (avg)_x", "Keywords (Visual X)_x", "Operators (Visual X)_x", "Keywords (Visual X)_y", "MIDQ (min)_x", "Operators (Visual X)_y", "#characters (max)_y", "NM (max)_x", "#identifiers (avg)_x", "Identifiers (Visual X)_x", "#identifiers (max)_x", "MIDQ (max)_y", "MIDQ (avg)_y", "Identifiers (Visual X)_y", "NMI (max)_y", "Line length (max)_y", "Cyclomatic complexity_x", "#identifiers (max)_y", "MIDQ (min)_y", "Cyclomatic complexity_y", "Line length (max)_x", "NMI (max)_x"]


ROOT_PATH="/home/nadeeshan/VPro2/complexity-verification-project/ml_model/src/ml-experiments"

DS3_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/data/final_features_ds3.csv"
DS3_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/data/merged_ds3.csv"

DS6_TASK1_PATH = ROOT_PATH + "/TASK1-AbsoluteTargets/Classifical-ML-Models/data/final_features_ds6.csv"
DS6_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/data/merged_ds6.csv"

MERGED_TASK2_PATH = ROOT_PATH + "/TASK2-RelativeComprehensibility/data/merged_ds.csv"

def load_data():
    ds3_task1_data = pd.read_csv(DS3_TASK1_PATH)
    ds3_task2_data = pd.read_csv(DS3_TASK2_PATH)

    ds6_task1_data = pd.read_csv(DS6_TASK1_PATH)
    ds6_task2_data = pd.read_csv(DS6_TASK2_PATH)

    merged_task2_data = pd.read_csv(MERGED_TASK2_PATH)

    return ds3_task1_data, ds3_task2_data, ds6_task1_data, ds6_task2_data, merged_task2_data

def get_feature_details(dataset, dataset_name, task):
    ds_data = dataset
    if task == "Task1":
        ds_features = task1_features
    else:
        ds_features = task2_features

    # csv_header = {"Task":"", "Dataset":"", "feature":"","min":"" , "max":"", "median":"", "avg":"", "std":"", "25th":"", "75th":"", "IQR":"", "Low wisker":"", "High wisker":"", "#Outliers":"", "Std scaler min":"", "Std scaler max":"", "Std scaler median":"", "Std scaler mean":"", "Std scaler 25%": "", "Std scaler 75%": "", "Std scaler IQR":"", "Std scaler Low wisker":"", "Std scaler High wisker":"", "Std scaler #Outliers":"", "Robust scaler min":"", "Robust scaler max":"", "Robust scaler median":"", "Robust scaler mean":"", "Robust scaler 25%": "", "Robust scaler 75%": "", "Robust scaler IQR":"", "Robust scaler Low wisker":"", "Robust scaler High wisker":"", "Robust scaler #Outliers":""}
    csv_header = {"Task":"", "Dataset":"", "feature":"","min":"" , "max":"", "median":"", "avg":"", "std":"", "25th":"", "75th":"", "IQR":"", "Low wisker":"", "High wisker":"", "#Outliers":"", "Std scaler min":"", "Std scaler max":"" ,"Robust scaler min":"", "Robust scaler max":""}
   
    for feature in ds_features:
        ## create a boxplot object and get above details
        csv_header["Task"] = task
        csv_header["Dataset"] = dataset_name
        csv_header["feature"] = feature
        csv_header["min"] = ds_data[feature].min()
        csv_header["max"] = ds_data[feature].max()
        csv_header["median"] = ds_data[feature].median()
        csv_header["avg"] = ds_data[feature].mean()
        csv_header["std"] = ds_data[feature].std()

        Q1 = ds_data[feature].quantile(0.25)
        Q3 = ds_data[feature].quantile(0.75)
        IQR = Q3 - Q1

        csv_header["25th"] = Q1
        csv_header["75th"] = Q3
        csv_header["IQR"] = IQR
        csv_header["Low wisker"] = Q1 - 1.5 * IQR
        csv_header["High wisker"] = Q3 + 1.5 * IQR

        ## count outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ds_data[(ds_data[feature] < lower_bound) | (ds_data[feature] > upper_bound)]
        csv_header["#Outliers"] = len(outliers)

        plt.figure()
        ## create the boxplots
        ds_data.boxplot(column=[feature], grid=False, showmeans=True)
        plt.savefig(ROOT_PATH + "/TASK2-RelativeComprehensibility/features/" + task + "/" + dataset_name + "/" + feature + ".png")
        plt.close()

        ## scale the feature
        std_scaler = StandardScaler()

        
        std = sorted(std_scaler.fit_transform(ds_data[feature].values.reshape(-1, 1)))
        csv_header["Std scaler min"] = std[0].item() if np.isnan(std[0].item()) == False else np.nan
        csv_header["Std scaler max"] = std[-1].item() if np.isnan(std[-1].item()) == False else np.nan

        robust_scaler = RobustScaler()
        rb = sorted(robust_scaler.fit_transform(ds_data[feature].values.reshape(-1, 1)))
        csv_header["Robust scaler min"] = rb[0].item() if np.isnan(rb[0].item()) == False else np.nan
        csv_header["Robust scaler max"] = rb[-1].item() if np.isnan(rb[-1].item()) == False else np.nan

        # csv_header["Std scaler median"] = np.median(std)
        # csv_header["Std scaler mean"] = np.mean(std)
        # csv_header["Std scaler 25%"] = np.percentile(std, 25)
        # csv_header["Std scaler 75%"] = np.percentile(std, 75)
        # csv_header["Std scaler IQR"] = np.percentile(std, 75) - np.percentile(std, 25)
        # csv_header["Std scaler Low wisker"] = np.percentile(std, 25) - 1.5 * (np.percentile(std, 75) - np.percentile(std, 25))
        # csv_header["Std scaler High wisker"] = np.percentile(std, 75) + 1.5 * (np.percentile(std, 75) - np.percentile(std, 25))
        # std_outliers = [x for x in std if x < np.percentile(std, 25) - 1.5 * (np.percentile(std, 75) - np.percentile(std, 25)) or x > np.percentile(std, 75) + 1.5 * (np.percentile(std, 75) - np.percentile(std, 25))]
        # csv_header["Std scaler #Outliers"] = len(std_outliers)

        # csv_header["Robust scaler median"] = np.median(rb)
        # csv_header["Robust scaler mean"] = np.mean(rb)
        # csv_header["Robust scaler 25%"] = np.percentile(rb, 25)
        # csv_header["Robust scaler 75%"] = np.percentile(rb, 75)
        # csv_header["Robust scaler IQR"] = np.percentile(rb, 75) - np.percentile(rb, 25)
        # csv_header["Robust scaler Low wisker"] = np.percentile(rb, 25) - 1.5 * (np.percentile(rb, 75) - np.percentile(rb, 25))
        # csv_header["Robust scaler High wisker"] = np.percentile(rb, 75) + 1.5 * (np.percentile(rb, 75) - np.percentile(rb, 25))
        # rb_outliers = [x for x in rb if x < np.percentile(rb, 25) - 1.5 * (np.percentile(rb, 75) - np.percentile(rb, 25)) or x > np.percentile(rb, 75) + 1.5 * (np.percentile(rb, 75) - np.percentile(rb, 25))]
        # csv_header["Robust scaler #Outliers"] = len(rb_outliers)


        ## if file exists, append it, if not create ##
        if os.path.exists(ROOT_PATH + "/TASK2-RelativeComprehensibility/features/feature_details.csv"):
            with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/features/feature_details.csv","a") as f:
                w = csv.DictWriter(f, csv_header.keys())
                w.writerow(csv_header)
        else:
            with open(ROOT_PATH + "/TASK2-RelativeComprehensibility/features/feature_details.csv", "w+") as f:
                w = csv.DictWriter(f, csv_header.keys())
                w.writeheader()
                w.writerow(csv_header)
        
def main():
    ds3_task1_data, ds3_task2_data, ds6_task1_data, ds6_task2_data, merged_task2_data = load_data()
    get_feature_details(ds3_task1_data, "DS3", "Task1")
    get_feature_details(ds3_task2_data, "DS3", "Task2")
    get_feature_details(ds6_task1_data, "DS6", "Task1")
    get_feature_details(ds6_task2_data, "DS6", "Task2")
    get_feature_details(merged_task2_data, "Merged", "Task2")

if __name__ == "__main__":
    main()    

    