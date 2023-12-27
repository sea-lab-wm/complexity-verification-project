'''
This file is used to generate the experiments for classification. 
'''
import json
import os

warning_features = ["warnings_checker_framework", "warnings_infer", "warnings_openjml", "warnings_typestate_checker", "warning_sum"]
classification_targets = ["PBU", "BD50", "ABU50"]

set_3_4 = {
    "feature-set3" : {
        "PBU": ["PE spec (java)", "MIDQ (max)", "CICsyn (max)", "NMI (avg)", "TC (avg)", "TC (max)", "Line length (max)", "LOC", "Line length (dft)", "Strings (area)"],
        "ABU50": ["PE spec (java)", "NMI (avg)", "TC (avg)", "#conditionals (avg)", "#periods (avg)", "Identifiers length (max)", "#comparisons (dft)", "Indentation length (dft)","Literals (Visual X)", "#parameters"],
        "BD50": ["PE spec (java)", "PE gen", "CR", "Cyclomatic complexity", "#assignments (avg)", "#words (max)", "#conditionals (dft)", "Numbers (Visual X)", "Literals (Visual Y)" , "#nested blocks (avg)"]
    },
    "feature-set4" : {
        "PBU": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"],
        "ABU50": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"],
        "BD50": ["developer_position", "PE gen", "PE spec (java)", "CR", "CIC (avg)",	"CIC (max)", "CICsyn (avg)", "CICsyn (max)", "MIDQ (min)","MIDQ (avg)", "MIDQ (max)", "AEDQ (min)", "AEDQ (avg)", "AEDQ (max)", "EAP (min)", "EAP (avg)", "EAP (max)", "Cyclomatic complexity", "IMSQ (min)", "IMSQ (avg)", "IMSQ (max)", "Readability",  "ITID (avg)", "NMI (avg)", "NMI (max)", "NM (avg)", "NM (max)", "TC (avg)", "TC (min)", "TC (max)", "#assignments (avg)", "#blank lines (avg)", "#commas (avg)", "#comments (avg)", "#comparisons (avg)",  "Identifiers length (avg)",  "#conditionals (avg)", "Indentation length (avg)", "#keywords (avg)", "Line length (avg)","#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)","#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)", "#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements"]
    }
}

ROOT_PATH='/verification_project/'

def generate_experiments(experiment_id, target, feature_set, features, smote):
    experiment = {
        "experiment_id": "exp" + str(experiment_id) + "-"+ target + "-" + feature_set + "-" + smote, 
        "features": features, 
        "target": target, 
        "use_SMOTE" : 'true' if smote == "SMOTE" else 'false'
    }
    return experiment

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

with open(ROOT_PATH + 'classification/experiments.jsonl', 'w+') as f:
    f.write('')

experiment_id = 0
for warning_feature in warning_features:
    for target in classification_targets:
        for smote in ['SMOTE', 'NO-SMOTE']:
            
            with open(ROOT_PATH + 'classification/experiments.jsonl', 'a') as f:
                
                experiment_id = experiment_id+1
                jsondump =json.dumps(generate_experiments(experiment_id, target, 'feature-set1', [warning_feature] + feature_set1[target], smote))
                ## remove "" from the use_SMOTE value in json string
                jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                f.write(jsondump)
                f.write('\n')

                experiment_id = experiment_id+1
                jsondump = json.dumps(generate_experiments(experiment_id, target, 'feature-set2', [warning_feature] + feature_set2[target], smote))
                jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                f.write(jsondump)
                f.write('\n')

                experiment_id = experiment_id+1
                jsondump = json.dumps(generate_experiments(experiment_id, target, 'feature-set3', [warning_feature] + set_3_4['feature-set3'][target], smote))
                jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                f.write(jsondump)
                f.write('\n')

                experiment_id = experiment_id+1
                jsondump = json.dumps(generate_experiments(experiment_id, target, 'feature-set4', [warning_feature] + set_3_4['feature-set4'][target], smote))
                jsondump = jsondump.replace('"true"', 'true').replace('"false"', 'false')
                f.write(jsondump)
                f.write('\n')
                
## remove the last new line character
with open(ROOT_PATH + 'classification/experiments.jsonl', 'rb+') as f:
    f.seek(-1, os.SEEK_END)
    f.truncate()                