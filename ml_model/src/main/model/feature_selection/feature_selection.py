import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif, RFE
from scipy.stats import kendalltau

import csv

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('ml_model/src/main/model/data/understandability_with_warnings.csv')
categorical_features = ["Cyclomatic complexity", "IMSQ (min)","MIDQ (min)","AEDQ (min)","EAP (min)","IMSQ (avg)", "MIDQ (avg)","AEDQ (avg)","EAP (avg)","IMSQ (max)","MIDQ (max)","AEDQ (max)", "EAP (max)","Readability", "ITID (avg)","NMI (avg)","NMI (max)","CIC (avg)","CIC (max)", "CICsyn (avg)", "CICsyn (max)", "CR", "NM (avg)", "NM (max)", "TC (avg)","TC (min)","TC (max)","#assignments (avg)","#blank lines (avg)","#commas (avg)","#comments (avg)","#comparisons (avg)", "Identifiers length (avg)", "#conditionals (avg)","Indentation length (avg)","#keywords (avg)","Line length (avg)", "#loops (avg)", "#identifiers (avg)", "#numbers (avg)","#operators (avg)", "#parenthesis (avg)", "#periods (avg)","#spaces (avg)","Identifiers length (max)","Indentation length (max)", "#keywords (max)", "Line length (max)",	"#identifiers (max)","#numbers (max)","#characters (max)","#words (max)","Entropy","Volume","LOC","#assignments (dft)","#commas (dft)","#comments (dft)", "#comparisons (dft)","#conditionals (dft)","Indentation length (dft)","#keywords (dft)","Line length (dft)","#loops (dft)","#identifiers (dft)","#numbers (dft)","#operators (dft)","#parenthesis (dft)","#periods (dft)","#spaces (dft)","Comments (Visual X)","Comments (Visual Y)","Identifiers (Visual X)","Identifiers (Visual Y)","Keywords (Visual X)","Keywords (Visual Y)","Numbers (Visual X)","Numbers (Visual Y)","Strings (Visual X)","Strings (Visual Y)","Literals (Visual X)","Literals (Visual Y)","Operators (Visual X)","Operators (Visual Y)","Comments (area)","Identifiers (area)","Keywords (area)","Numbers (area)","Strings (area)","Literals (area)","Operators (area)",	"Keywords/identifiers (area)","Numbers/identifiers (area)","Strings/identifiers (area)","Literals/identifiers (area)","Operators/literals (area)","Numbers/keywords (area)","Strings/keywords (area)","Literals/keywords (area)","Operators/keywords (area)", "#aligned blocks","Extent of aligned blocks","#nested blocks (avg)","#parameters","#statements", "warnings_checker_framework", "warnings_typestate_checker", "warnings_infer"	, "warnings_openjml",	"warning_sum"]
categorical_target = ["PBU", "BD50", "ABU50"]
X = df[categorical_features]
y = df[categorical_target]

stat_header = {'input feature':'', 'output_feature': '', 'tau': 0, 'p-value':0}


######################################
## Classification feature selection ##
######################################
## Numeric input, categorical output

## FILTER METHODS ##
## write header
with open('ml_model/src/main/model/feature_selection/kendall_features.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
    writer.writeheader()
## Kendals Tau
for feature in categorical_features:
    for target in categorical_target:
        tau,p = kendalltau(X[feature], y[target])
        # write to csv
        with open('ml_model/src/main/model/feature_selection/kendall_features.csv', "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
            writer.writerow({'input feature':feature, 'output_feature': target, 'tau': tau, 'p-value':p})
## interprere the results
# if p-value < 0.05, then the correlation is statistically significant. means that there is a relationship between the two variables.
# filter based on above criteria and sort tau in descending order. now we have the most important features to predict the target variable.

stat_header = {'input feature':'', 'output_feature': ''}
## write header
with open('ml_model/src/main/model/feature_selection/anova_features.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
    writer.writeheader()
## ANOVA f-test
for target in categorical_target:
    selector = SelectKBest(k=10, score_func=f_classif)
    selector.fit_transform(X, y[target])
    mask = selector.get_support()
    new_features = [] # The list of your K best features
    feature_names = list(X.columns.values)
    for bool_val, feature in zip(mask, feature_names):
        if bool_val:
            new_features.append(feature)
    # write to csv
    with open('ml_model/src/main/model/feature_selection/anova_features.csv', "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writerow({'input feature':new_features, 'output_feature': target})  

stat_header = {'input feature':'', 'output_feature': ''}
## write header
with open('ml_model/src/main/model/feature_selection/mutual_infor_features.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
    writer.writeheader()
## Mutual information classification    
for target in categorical_target:
    feature_scores = mutual_info_classif(X, y[target], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)   
    threshold = 10
    high_score_features = []
    for score, f_name in sorted(zip(feature_scores, X.columns), reverse=True)[:threshold]:
        high_score_features.append(f_name)
    # write to csv
    with open('ml_model/src/main/model/feature_selection/mutual_infor_features.csv', "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writerow({'input feature':high_score_features, 'output_feature': target})        

## SelectFromModel
stat_header = {'input feature':'', 'output_feature': ''}
## write header
with open('ml_model/src/main/model/feature_selection/sfm_features.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
    writer.writeheader()
for target in categorical_target:
    model_lr = LogisticRegression(C=0.01, random_state=0)
    model_lr.fit(X, y[target])
    selector = SelectFromModel(model_lr, threshold='2*median', prefit=True)
    mask = selector.get_support()
    sfm = X.iloc[:, mask]
    #write to csv
    with open('ml_model/src/main/model/feature_selection/sfm_features.csv', "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writerow({'input feature':sfm.columns.values, 'output_feature': target})


## Wrapper feature selection ##
# Recursive feature elimination
stat_header = {'input feature':'', 'output_feature': ''}
## write header
with open('ml_model/src/main/model/feature_selection/rfe_features.csv', "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
    writer.writeheader()
for target in categorical_target:
    threshold = 7 # the number of most relevant features to select
    estimator = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0)
    selector = RFE(estimator, n_features_to_select=threshold, step=1)
    selector = selector.fit(X, y[target])
    mask = selector.get_support()
    rfe = X.iloc[:, mask]
    #write to csv
    with open('ml_model/src/main/model/feature_selection/rfe_features.csv', "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writerow({'input feature':rfe.columns.values, 'output_feature': target})

