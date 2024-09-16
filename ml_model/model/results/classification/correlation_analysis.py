'''
This file is used to perform Kendall's Tau correlation analysis on 
selected features with target variable.
Input:  selected features, target variable
Output: Kendall's Tau correlation coefficient for each feature with target variable
'''
import pandas as pd
import csv
from scipy.stats import kendalltau

feature_sets = ["set1", "set2", "set3", "set4"]
classification_targets = ["PBU", "BD50", "ABU50"]

ROOT_PATH="/verification_project/" 

csv_header = ["feature_set", "feature", "Target", "Kendall's Tau", "p-value"]


def read_data(file_name):
    '''
    This function is used to read the data from the file and output {taget: [<features>]}.
    Sample input: file_name = 'final_features1_bfs.txt'
    Sample output: {'PBU': ['feature1', 'feature2', 'feature3'],
                    'ABU50': ['feature1', 'feature2', 'feature3'], 
                    'BD50': ['feature1', 'feature2', 'feature3']}
    '''
    data = {}
    with open(file_name) as f:
        for line, target in zip(f, classification_targets):
            ## remove the new line character
            line = line.strip()
            data[target] = line.split(target)[1]
            ## convert this string to list'["Test", "Test2"]' --> ["Test", "Test2"]
            data[target] = data[target].replace('[', '').replace(']', '').replace('"', '').split(',')
            ## remove the extra spaces in each item in the list
            data[target] = [item.strip() for item in data[target]]
    return data

## read the data for the feature set1
feature_set1 = read_data(ROOT_PATH + 'feature_selection/classification/final_features1_bfs.txt')

## read the data for the feature set2
feature_set2 = read_data(ROOT_PATH + 'feature_selection/classification/final_features2_bfs.txt')




all_sets = {
    ################################################################################
    ###### Replace the features with the selected features for each feature set#####
    ################################################################################
    # Feature Set 1: refer to feature_selection/classification/final_features1_bfs.txt
    "set1" : feature_set1,

    # Replace the features with the selected features for each feature set
    # Feature Set 2: refer to feature_selection/classification/final_features2_bfs.txt
    "set2" : feature_set2,

    # These are the features for feature set 3. Extracted from the Replication package 
    "set3" : {
        "PBU": ["PE spec (java)", "MIDQ (max)", "CICsyn (max)", "NMI (avg)", "TC (avg)", "TC (max)", "Line length (max)", "LOC", "Line length (dft)", "Strings (area)"],
        "ABU50": ["PE spec (java)", "NMI (avg)", "TC (avg)", "#conditionals (avg)", "#periods (avg)", "Identifiers length (max)", "#comparisons (dft)", "Indentation length (dft)","Literals (Visual X)", "#parameters"],
        "BD50": ["PE spec (java)", "PE gen", "CR", "Cyclomatic complexity", "#assignments (avg)", "#words (max)", "#conditionals (dft)", "Numbers (Visual X)", "Literals (Visual Y)" , "#nested blocks (avg)"]
    },
    "set4" : {
        "PBU": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"],
        "ABU50": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"],
        "BD50": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"]
    }
}

## write csv header
with open(ROOT_PATH + "Results/classification/correlation_analysis.csv", "w+") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(csv_header)

## perform Kendall's Tau correlation analysis
## on each feature set with target variable
for target in classification_targets:
    for feature_set in feature_sets:
        all_feature_with_taget = all_sets[feature_set][target] + [target]
        df = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_" + target + ".csv")[all_feature_with_taget]
        if feature_set == "set3":
            ## this is required because for feature set 3 we removed the class overalp instances by removing rows while for others we add 3 developer features to remove class overlap
            df = pd.read_csv(ROOT_PATH + "data/understandability_with_warnings_" + target + "_set3.csv")[all_feature_with_taget]
        
        ## compute correlation coefficient 
        for feature in all_sets[feature_set][target]:
            tau, p_value = kendalltau(df[feature], df[target], nan_policy='omit')
            with open(ROOT_PATH + "Results/classification/correlation_analysis.csv", "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([feature_set, feature, target, tau, p_value])  
