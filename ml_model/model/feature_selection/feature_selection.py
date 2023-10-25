'''
This feature selection based on the paper https://ieeexplore.ieee.org/document/8651396

## Feature selection procedure ##
## 1. Perform Kendall's Tau correlation test between each pair of features.
## 2. Pick the pairs that show |Tau| >= 0.7 (highly correlated)
## 3. Remove one of the features which has the highest number of missing values.
## 4. If missing values are equal, remove one of the features randomly.
2. Remove Keywords/Comments (area) because it has the highest number of missing values
'''
import pandas as pd
import csv
import json

from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/data/understandability_with_warnings.csv')

## all code features 
features = [
    "Cyclomatic complexity", 
    "IMSQ (min)", 
    "IMSQ (avg)", 
    "IMSQ (max)",
    "Readability", 
    "ITID (avg)",
    "NMI (avg)",
    "NMI (max)",
    "NM (avg)", 
    "NM (max)", 
    "TC (avg)",
    "TC (min)",
    "TC (max)",
    "#assignments (avg)",
    "#blank lines (avg)",
    "#commas (avg)",
    "#comments (avg)",
    "#comparisons (avg)", 
    "Identifiers length (avg)", 
    "#conditionals (avg)",
    "Indentation length (avg)",
    "#keywords (avg)",
    "Line length (avg)",
    "#loops (avg)", 
    "#identifiers (avg)", 
    "#numbers (avg)",
    "#operators (avg)", 
    "#parenthesis (avg)", 
    "#periods (avg)",
    "#spaces (avg)",
    "Identifiers length (max)",
    "Indentation length (max)", 
    "#keywords (max)", 
    "Line length (max)",
    "#identifiers (max)",
    "#numbers (max)",
    "#characters (max)",
    "#words (max)",
    "Entropy",
    "Volume",
    "LOC",
    "#assignments (dft)",
    "#commas (dft)",
    "#comments (dft)", 
    "#comparisons (dft)",
    "#conditionals (dft)",
    "Indentation length (dft)",
    "#keywords (dft)",
    "Line length (dft)",
    "#loops (dft)",
    "#identifiers (dft)",
    "#numbers (dft)",
    "#operators (dft)",
    "#parenthesis (dft)",
    "#periods (dft)",
    "#spaces (dft)",
    "Comments (Visual X)",
    "Comments (Visual Y)",
    "Identifiers (Visual X)",
    "Identifiers (Visual Y)",
    "Keywords (Visual X)",
    "Keywords (Visual Y)",
    "Numbers (Visual X)",
    "Numbers (Visual Y)",
    "Strings (Visual X)",
    "Strings (Visual Y)",
    "Literals (Visual X)",
    "Literals (Visual Y)",
    "Operators (Visual X)",
    "Operators (Visual Y)",
    "Comments (area)",
    "Identifiers (area)",
    "Keywords (area)",
    "Numbers (area)",
    "Strings (area)",
    "Literals (area)",
    "Operators (area)",	
    "Keywords/identifiers (area)",
    "Numbers/identifiers (area)",
    "Strings/identifiers (area)",
    "Literals/identifiers (area)",
    "Operators/literals (area)",
    "Numbers/keywords (area)",
    "Strings/keywords (area)",
    "Literals/keywords (area)",
    "Operators/keywords (area)", 
    "#aligned blocks",
    "Extent of aligned blocks",
    "#nested blocks (avg)",
    "#parameters",
    "#statements", 
    "Strings/comments (area)",
    "Keywords/comments (area)",
    "Literals/comments (area)",
    "Strings/numbers (area)",
    "Numbers/comments (area)",
    "Operators/numbers (area)",
    "Literals/numbers (area)",
    "Identifiers/comments (area)",
    "Literals/strings (area)",
    "Operators/strings (area)",
    "Operators/comments (area)",
    "Operators/literals (area).1"
]
## Feature set 1: Consider only the code features. Took Highly correlated features. Removed one of highly correlated ones which has the larget missing values or feature1
## perfom SFFS to select the best features
## final_features1.txt

## Feature set 2: Consider only the code features. Took Highly correlated features. Removed one of highly correlated ones which has the larget missing values or feature2
## perfom SFFS to select the best features
## final_features1.txt

## Feeature set 3 Consider only code features. Standard Features from the paper in the replication package.
## remove below features because they are not code features.
## PBU - PE spec (java), MIDQ (max), CICsyn (max), 
## ABU50 - PE spec (java)
## BD50 - PE spec (java), PE gen, CR

## Features:
## PBU - "NMI (avg)", "TC (avg)", "TC (max)", "Line length (max)", "LOC", "Line length (dft)", "Strings (area)"
## ABU50 - "NMI (avg)", "TC (avg)", "#conditionals (avg)", "#periods (avg)", "Identifiers length (max)", "#comparisons (dft)", "Indentation length (dft)","Literals (Visual X)", "#parameters"
## BD50 - "Cyclomatic complexity", "#assignments (avg)", "#words (max)", "#conditionals (dft)", "Numbers (Visual X)", "Literals (Visual Y)" , "#nested blocks (avg)"


## Feature set 4 consider all code features. Remove below features because they have the highest number of missing values.
## "Operators/literals (area).1", "Operators/comments (area)", "Identifiers/comments (area)", "Literals/numbers (area)", "Operators/numbers (area)", Numbers/comments (area), Strings/numbers (area), Literals/comments (area), Keywords/comments (area), Strings/comments (area), Operators/strings (area),
## Literals/strings (area)

warning_features = ["warnings_checker_framework", "warnings_typestate_checker", "warnings_infer", "warnings_openjml", "warning_sum"]
categorical_target = ["PBU", "BD50", "ABU50"]
X = df[features]
y = df[categorical_target]

stat_header = {'feature1':'', 'feature2': '', 'tau': 0, 'p-value':0, 'missing values in feature1':0, 'missing values in feature2':0, 'feature to remove':''}

ROOT_PATH = '/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/'

######################################
## Feature selection ##
######################################

def kendals_feature_selection(features, featureToRemove, output_file):
    ## write header
    with open(ROOT_PATH + 'feature_selection/' + output_file, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
        writer.writeheader()

    for feature1 in features:
        for feature2 in features[features.index(feature1)+1:]:
            tau,p = kendalltau(X[feature1], X[feature2], nan_policy='omit') ## only perform pair-wise comparisions where both the data points has values (i.e not NAN)
            if abs(tau) >= 0.7 and feature1 != feature2:
                ## number of data points with missing values
                feature1_missing = X[feature1].isna().sum()
                feature2_missing = X[feature2].isna().sum()
                
                ## remove one of the features which has the highest number of missing values.
                if feature1_missing > feature2_missing:
                    feature_to_remove = feature1
                elif feature1_missing < feature2_missing:
                    feature_to_remove = feature2
                else:
                    feature_to_remove = feature1 if featureToRemove == "feature1" else feature2
                # write to csv
                with open(ROOT_PATH + 'feature_selection/' + output_file, "a") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
                    writer.writerow({'feature1':feature1, 'feature2': feature2, 'tau': tau, 'p-value': p, 'missing values in feature1':feature1_missing, 'missing values in feature2':feature2_missing, 'feature to remove': feature_to_remove})


def linear_floating_forward_feature_selection(df_features_X, df_target_y, kFold):
    '''
    Perform linear floating forward selection with wrapper strategy 
    We select logistic regression as the wrapper
    We evaluate the performance of the model using cross validation and F1 as the metric
    Sequential Forward Floating Selection
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-2-toggling-between-sfs-sbs-sffs-and-sbfs
    '''
    print('\nSequential Forward Floating Feature Selection (k=10)...')
    lr = LogisticRegression()
    sffs = SFS(lr, forward=True, floating=True, scoring='f1', cv=kFold, n_jobs=-1)
    sffs = sffs.fit(df_features_X, df_target_y.to_numpy().ravel())
    
    return list(sffs.k_feature_names_)

## STEP 1 ##
## Remove redundent features
## How we do it:
## 1. Perform Kendall's Tau correlation test between each pair of features.
## 2. Pick the pairs that show |Tau| >= 0.7 (highly correlated)
## 3. Remove one of the features which has the highest number of missing values.
## 4. If missing values are equal, remove one of the features randomly.
kendals_feature_selection(features, "feature1", "kendall_highly_correlated_features_part1.csv") ## remove the first feature in the pair
kendals_feature_selection(features, "feature2", "kendall_highly_correlated_features_part2.csv") ## remove the second feature in the pair


## STEP 2 ##
## Remove Keywords/Comments (area) because it has the highest number of missing values
## and create the final feature set
def remove_specific_feature(input_feature_file):
    df_remove_fea = pd.read_csv(ROOT_PATH + 'feature_selection/' + input_feature_file)
    df_remove_fea = df_remove_fea['feature to remove']
    features_to_remove = list(set(df_remove_fea.to_list())) 
    features_to_remove.append('Keywords/comments (area)') ## append the specific feature to remove
    final_list_of_features = list(set(features) - set(features_to_remove))

    return final_list_of_features        

features_part1 = remove_specific_feature('kendall_highly_correlated_features_part1.csv')
features_part2 = remove_specific_feature('kendall_highly_correlated_features_part2.csv')


## STEP 3 ##
## Perform linear floating forward feature selection
## We select logistic regression as the wrapper
## We evaluate the performance of the model using 5-fold cross validation and F1 as the metric
## Sequential Forward Floating Selection
## https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-2-toggling-between-sfs-sbs-sffs-and-sbfs
## StratifiedKFold
Folds = 5
RANDOM_SEED = 42
kFold = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=RANDOM_SEED)
with open(ROOT_PATH + 'feature_selection/final_features1.txt','w+') as file1:
    file1.write('')
with open(ROOT_PATH + 'feature_selection/final_features2.txt','w+') as file2:
    file2.write('')

for target in categorical_target:
    final_features1 = linear_floating_forward_feature_selection(X[features_part1], y[target], kFold)
    final_features2 = linear_floating_forward_feature_selection(X[features_part2], y[target], kFold)
    
    ## write to text file
    with open(ROOT_PATH + 'feature_selection/final_features1.txt','a') as file1:
        file1.write(target + json.dumps(final_features1) + '\n')
    with open(ROOT_PATH + 'feature_selection/final_features2.txt','a') as file2:
        file2.write(target + json.dumps(final_features2) + '\n')    

## Kendals Tau
# for feature in categorical_features:
#     for target in categorical_target:
#         tau,p = kendalltau(X[feature], y[target])
#         # write to csv
#         with open('ml_model/src/main/model/feature_selection/kendall_features.csv', "a") as csv_file:
#             writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#             writer.writerow({'input feature':feature, 'output_feature': target, 'tau': tau, 'p-value':p})
## interprere the results
# if p-value < 0.05, then the correlation is statistically significant. means that there is a relationship between the two variables.
# filter based on above criteria and sort tau in descending order. now we have the most important features to predict the target variable.

# stat_header = {'input feature':'', 'output_feature': ''}
# ## write header
# with open('ml_model/src/main/model/feature_selection/anova_features.csv', "w") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#     writer.writeheader()
# ## ANOVA f-test
# for target in categorical_target:
#     selector = SelectKBest(k=10, score_func=f_classif)
#     selector.fit_transform(X, y[target])
#     mask = selector.get_support()
#     new_features = [] # The list of your K best features
#     feature_names = list(X.columns.values)
#     for bool_val, feature in zip(mask, feature_names):
#         if bool_val:
#             new_features.append(feature)
#     # write to csv
#     with open('ml_model/src/main/model/feature_selection/anova_features.csv', "a") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#         writer.writerow({'input feature':new_features, 'output_feature': target})  

# stat_header = {'input feature':'', 'output_feature': ''}
# ## write header
# with open('ml_model/src/main/model/feature_selection/mutual_infor_features.csv', "w") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#     writer.writeheader()
# ## Mutual information classification    
# for target in categorical_target:
#     feature_scores = mutual_info_classif(X, y[target], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)   
#     threshold = 10
#     high_score_features = []
#     for score, f_name in sorted(zip(feature_scores, X.columns), reverse=True)[:threshold]:
#         high_score_features.append(f_name)
#     # write to csv
#     with open('ml_model/src/main/model/feature_selection/mutual_infor_features.csv', "a") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#         writer.writerow({'input feature':high_score_features, 'output_feature': target})        

## SelectFromModel
# stat_header = {'input feature':'', 'output_feature': ''}
# ## write header
# with open('ml_model/src/main/model/feature_selection/sfm_features.csv', "w") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#     writer.writeheader()
# for target in categorical_target:
#     model_lr = LogisticRegression(C=0.01, random_state=0)
#     model_lr.fit(X, y[target])
#     selector = SelectFromModel(model_lr, threshold='2*median', prefit=True)
#     mask = selector.get_support()
#     sfm = X.iloc[:, mask]
#     #write to csv
#     with open('ml_model/src/main/model/feature_selection/sfm_features.csv', "a") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#         writer.writerow({'input feature':sfm.columns.values, 'output_feature': target})


## Wrapper feature selection ##
# Recursive feature elimination
# stat_header = {'input feature':'', 'output_feature': ''}
# ## write header
# with open('ml_model/src/main/model/feature_selection/rfe_features.csv', "w") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#     writer.writeheader()
# for target in categorical_target:
#     threshold = 7 # the number of most relevant features to select
#     estimator = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0)
#     selector = RFE(estimator, n_features_to_select=threshold, step=1)
#     selector = selector.fit(X, y[target])
#     mask = selector.get_support()
#     rfe = X.iloc[:, mask]
#     #write to csv
#     with open('ml_model/src/main/model/feature_selection/rfe_features.csv', "a") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=stat_header.keys())
#         writer.writerow({'input feature':rfe.columns.values, 'output_feature': target})

